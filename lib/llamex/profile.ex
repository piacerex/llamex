defmodule Llamex.Profile do
  @moduledoc """
  Small profiling helpers for local GGUF generation experiments.
  """

  alias Llamex.{Context, ContextWindow, PreparedModel, Tensor}
  alias Llamex.Layers.{Attention, Linear}

  def timed(label, fun) when is_binary(label) and is_function(fun, 0) do
    {microseconds, result} = :timer.tc(fun)

    {%{label: label, milliseconds: div(microseconds, 1000)}, result}
  end

  def generation_step(model, prompt, opts) when is_binary(prompt) and is_map(opts) do
    backend = profile_backend(model, opts)
    display_model = profile_model(model)
    reset_profile_caches(backend)
    sampler = sampler(opts)
    candidate_count = Map.get(opts, :candidate_count, 0)

    {prefill_time, {state, prefill_timings}} = timed_prefill(model, prompt, backend, opts)

    {step_time, step} =
      timed("step", fn ->
        timed_step(state.context, state.current_token, %{
          sampler: sampler,
          history: state.prompt_tokens,
          candidate_count: candidate_count
        })
      end)

    %{
      backend: step.context.backend,
      exla: exla_info(step.context.backend),
      backend_profile: backend_profile(state.context),
      prompt_tokens: length(state.prompt_tokens),
      prompt_token_ids: state.prompt_tokens,
      prompt_pieces: token_pieces(display_model, state.prompt_tokens),
      original_prompt_token_count: state.original_prompt_token_count,
      context_window: state.context_window,
      prompt_truncated?: state.prompt_truncated?,
      prepared?: state.prepared?,
      token: step.token,
      text: step.text,
      candidates: token_candidates(display_model, step.candidates),
      eval_timings: step.eval_timings,
      prefill_timings: prefill_timings,
      prompt_eval_steps: state.prompt_eval_steps,
      prompt_eval_summary: state.prompt_eval_summary,
      timings: [prefill_time, step_time],
      timing_summary:
        timing_summary([prefill_time, step_time], prefill_timings, [step.eval_timings])
    }
  end

  def prefill_steps(model, prompt, opts) when is_binary(prompt) and is_map(opts) do
    backend = profile_backend(model, opts)
    reset_profile_caches(backend)
    display_model = profile_model(model)
    original_prompt_tokens = Llamex.encode(display_model, prompt)
    context_window = ContextWindow.resolve(display_model, opts)
    prompt_tokens = ContextWindow.apply(original_prompt_tokens, context_window)
    context = profile_context(model, backend)
    prefill_tokens = Enum.drop(prompt_tokens, -1)

    {steps, context} =
      prefill_tokens
      |> Enum.with_index(1)
      |> Enum.reduce({[], context}, fn {token, index}, {steps, context} ->
        {timing, {context, _logits}} =
          timed("prefill_#{index}", fn ->
            Llamex.Engine.eval(context, token)
          end)

        step = %{
          index: index,
          token: token,
          piece: Map.fetch!(display_model.tokenizer.id_to_token, token),
          timing: timing
        }

        {[step | steps], context}
      end)

    %{
      prompt_tokens: prompt_tokens,
      original_prompt_token_count: length(original_prompt_tokens),
      context_window: context_window,
      prompt_truncated?: length(prompt_tokens) < length(original_prompt_tokens),
      current_token: List.last(prompt_tokens),
      current_piece: Map.fetch!(display_model.tokenizer.id_to_token, List.last(prompt_tokens)),
      context_tokens: context.tokens,
      steps: Enum.reverse(steps)
    }
  end

  def generation_steps(model, prompt, opts) when is_binary(prompt) and is_map(opts) do
    backend = profile_backend(model, opts)
    display_model = profile_model(model)
    reset_profile_caches(backend)
    sampler = sampler(opts)
    max_new_tokens = Llamex.MaxNewTokens.get(opts, 1)
    stop_tokens = stop_tokens(opts)
    stop_sequences = stop_sequences(opts)
    candidate_count = Map.get(opts, :candidate_count, 0)

    {prefill_time, {state, prefill_timings}} = timed_prefill(model, prompt, backend, opts)

    effective_max_new_tokens =
      ContextWindow.generation_budget(
        max_new_tokens,
        length(state.prompt_tokens),
        state.context_window
      )

    {steps, context, _current_token, _sampler_state, finish_reason} =
      Enum.reduce_while(
        step_indexes(effective_max_new_tokens),
        {[], state.context, state.current_token, nil, :length},
        fn index, {steps, context, current_token, sampler_state, _finish_reason} ->
          history = state.prompt_tokens ++ generated_tokens_from_acc(steps)

          {step_time, step} =
            timed("step_#{index}", fn ->
              timed_step(context, current_token, %{
                sampler: sampler,
                sampler_state: sampler_state,
                history: history,
                candidate_count: candidate_count
              })
            end)

          step_info =
            display_model
            |> token_info(step.token)
            |> Map.merge(%{
              index: index,
              text: step.text,
              timing: step_time,
              eval_timings: step.eval_timings,
              candidates: token_candidates(display_model, step.candidates)
            })

          finish_reason = if stop_token?(step.token, stop_tokens), do: :stop, else: :length

          generated_text =
            Llamex.decode(display_model, generated_tokens_from_acc([step_info | steps]))

          finish_reason =
            if finish_reason == :stop or not stop_sequence?(generated_text, stop_sequences) do
              finish_reason
            else
              :stop_sequence
            end

          next_state =
            {[step_info | steps], step.context, step.token, step.sampler_state, finish_reason}

          if finish_reason in [:stop, :stop_sequence] do
            {:halt, next_state}
          else
            {:cont, next_state}
          end
        end
      )

    steps = Enum.reverse(steps)
    generated_tokens = Enum.map(steps, & &1.token)

    %{
      backend: context.backend,
      exla: exla_info(context.backend),
      backend_profile: backend_profile(context),
      max_new_tokens: max_new_tokens,
      requested_max_new_tokens: max_new_tokens,
      effective_max_new_tokens: effective_max_new_tokens,
      stop_token: List.first(stop_tokens),
      stop_tokens: stop_tokens,
      stop_sequences: stop_sequences,
      sampler: display_sampler(sampler),
      prompt_tokens: length(state.prompt_tokens),
      original_prompt_token_count: state.original_prompt_token_count,
      context_window: state.context_window,
      prompt_truncated?: state.prompt_truncated?,
      prepared?: state.prepared?,
      prompt_token_ids: state.prompt_tokens,
      prompt_pieces: token_pieces(display_model, state.prompt_tokens),
      generated_tokens: generated_tokens,
      generated_pieces: token_pieces(display_model, generated_tokens),
      generated_token_info: Enum.map(generated_tokens, &token_info(display_model, &1)),
      finish_reason: finish_reason(finish_reason, max_new_tokens, effective_max_new_tokens),
      text: Llamex.decode(display_model, generated_tokens),
      prefill_timings: prefill_timings,
      prompt_eval_steps: state.prompt_eval_steps,
      prompt_eval_summary: state.prompt_eval_summary,
      timings: [prefill_time | Enum.map(steps, & &1.timing)],
      timing_summary:
        timing_summary(
          [prefill_time | Enum.map(steps, & &1.timing)],
          prefill_timings,
          Enum.map(steps, & &1.eval_timings)
        ),
      steps: steps
    }
  end

  defp timing_summary(timings, prefill_timings, eval_timings) do
    components = component_summary(prefill_timings, eval_timings)

    %{
      total_milliseconds: sum_milliseconds(timings),
      prefill_milliseconds: sum_milliseconds(prefill_timings),
      step_milliseconds: timings |> Enum.reject(&(&1.label == "prefill")) |> sum_milliseconds(),
      eval_milliseconds: sum_eval_milliseconds(eval_timings),
      components: components,
      top_components: top_timing_labels(components),
      top_layers: top_timing_labels(components, &layer_timing_label?/1)
    }
  end

  defp component_summary(prefill_timings, eval_timings) do
    prefill =
      prefill_timings
      |> Enum.map(fn timing -> {"prefill.#{timing.label}", timing.milliseconds} end)

    eval =
      eval_timings
      |> Enum.flat_map(&flatten_eval_timing/1)

    (prefill ++ eval)
    |> Enum.group_by(fn {label, _milliseconds} -> label end, fn {_label, milliseconds} ->
      milliseconds
    end)
    |> Map.new(fn {label, values} -> {label, Enum.sum(values)} end)
  end

  defp flatten_eval_timing(%{layers: layers, output_norm: output_norm, logits: logits}) do
    [
      {"eval.output_norm", output_norm.milliseconds},
      {"eval.#{logits.label}", logits.milliseconds}
    ] ++
      Enum.flat_map(layers, &flatten_layer_timing/1)
  end

  defp flatten_layer_timing(layer) do
    [{"eval.#{layer.label}", layer.milliseconds}] ++
      Enum.flat_map(Map.get(layer, :components, []), fn component ->
        flatten_component_timing("eval.#{layer.label}.#{component.label}", component)
      end)
  end

  defp flatten_component_timing(prefix, %{components: components} = timing) do
    [{prefix, timing.milliseconds}] ++
      Enum.flat_map(components, fn component ->
        flatten_component_timing("#{prefix}.#{component.label}", component)
      end)
  end

  defp flatten_component_timing(prefix, timing), do: [{prefix, timing.milliseconds}]

  defp sum_eval_milliseconds(eval_timings) do
    eval_timings
    |> Enum.flat_map(&flatten_eval_timing/1)
    |> Enum.map(fn {_label, milliseconds} -> milliseconds end)
    |> Enum.sum()
  end

  defp sum_milliseconds(timings) do
    timings
    |> Enum.map(& &1.milliseconds)
    |> Enum.sum()
  end

  defp prompt_eval_summary(steps) do
    flattened =
      steps
      |> Enum.flat_map(&flatten_eval_timing(&1.eval_timings))

    %{
      token_count: length(steps),
      total_milliseconds:
        flattened |> Enum.map(fn {_label, milliseconds} -> milliseconds end) |> Enum.sum(),
      layers: timing_label_summary(flattened, &layer_timing_label?/1),
      components: timing_label_summary(flattened, fn _label -> true end)
    }
  end

  defp timing_label_summary(flattened, predicate) do
    flattened
    |> Enum.filter(fn {label, _milliseconds} -> predicate.(label) end)
    |> Enum.group_by(fn {label, _milliseconds} -> label end, fn {_label, milliseconds} ->
      milliseconds
    end)
    |> Enum.map(fn {label, milliseconds} ->
      %{label: label, milliseconds: Enum.sum(milliseconds)}
    end)
    |> Enum.sort_by(& &1.label)
  end

  defp top_timing_labels(components, predicate \\ fn _label -> true end) do
    components
    |> Enum.filter(fn {label, _milliseconds} -> predicate.(label) end)
    |> Enum.map(fn {label, milliseconds} -> %{label: label, milliseconds: milliseconds} end)
    |> Enum.sort_by(&{-&1.milliseconds, &1.label})
    |> Enum.take(10)
  end

  defp layer_timing_label?(label), do: Regex.match?(~r/^eval\.layer_\d+$/, label)

  defp generated_tokens_from_acc(steps) do
    steps
    |> Enum.reverse()
    |> Enum.map(& &1.token)
  end

  defp exla_info(backend) when backend in [Llamex.Backend.Nx, Llamex.Backend.NxEXLA] do
    Llamex.Backend.NxEXLA.configured()
  end

  defp exla_info(_backend), do: nil

  defp reset_profile_caches(backend) when backend in [Llamex.Backend.Nx, Llamex.Backend.NxEXLA] do
    Llamex.Backend.NxEXLA.clear_process_caches()
  end

  defp reset_profile_caches(_backend), do: :ok

  defp backend_profile(%Context{backend: backend, model: model} = context)
       when backend in [Llamex.Backend.Nx, Llamex.Backend.NxEXLA] do
    layer_count = length(model.layers)

    %{
      tensor_backend?: true,
      layer_count: layer_count,
      qkv_combined_layers: count_layers(model.layers, :w_qkv),
      gate_up_combined_layers: count_layers(model.layers, :w_gate_up),
      tensor_attention_norm_layers: count_layers(model.layers, :attention_norm),
      tensor_feed_forward_norm_layers: count_layers(model.layers, :feed_forward_norm),
      output_norm_tensor?: tensor?(model.output_norm),
      output_weight_tensor?: tensor?(get_in(model.output, [:weight])),
      prepared_kv_cache_entries: map_size(context.kv_cache.prepared_layers),
      nx_exla_cache: Llamex.Backend.NxEXLA.cache_stats(),
      nx_exla_prepare: Llamex.Backend.NxEXLA.prepare_stats()
    }
  end

  defp backend_profile(%Context{model: model}) do
    %{
      tensor_backend?: false,
      layer_count: length(model.layers)
    }
  end

  defp count_layers(layers, key) do
    Enum.count(layers, fn layer -> tensor?(Map.get(layer, key)) end)
  end

  defp tensor?(%Nx.Tensor{}), do: true
  defp tensor?(_value), do: false

  defp step_indexes(0), do: []
  defp step_indexes(max_new_tokens), do: 1..max_new_tokens

  defp finish_reason(:length, requested_max_new_tokens, effective_max_new_tokens) do
    if ContextWindow.context_limited?(requested_max_new_tokens, effective_max_new_tokens) do
      :context_window
    else
      :length
    end
  end

  defp finish_reason(reason, _requested_max_new_tokens, _effective_max_new_tokens), do: reason

  defp display_sampler(%{suppress_tokens: suppress_tokens} = sampler)
       when is_list(suppress_tokens) do
    sampler
    |> Map.delete(:suppress_tokens)
    |> Map.put(:suppressed_token_count, length(suppress_tokens))
  end

  defp display_sampler(sampler) when is_map(sampler), do: sampler
  defp display_sampler(sampler), do: sampler

  defp timed_prefill(model, prompt, backend, opts) do
    timed("prefill", fn ->
      source_model = model
      model = profile_model(source_model)
      context_window = ContextWindow.resolve(model, opts)

      {encode_time, original_prompt_tokens} =
        timed("prompt_encode", fn ->
          Llamex.encode(model, prompt)
        end)

      prompt_tokens = ContextWindow.apply(original_prompt_tokens, context_window)

      {prepare_time, context} =
        timed("backend_prepare", fn ->
          case Map.fetch(opts, :prepared_model) do
            {:ok, prepared_model} -> Llamex.Context.new_prepared(prepared_model, backend)
            :error -> profile_context(source_model, backend)
          end
        end)

      {prompt_eval_time, {context, prompt_eval_steps}} =
        timed("prompt_eval", fn ->
          prompt_tokens
          |> Enum.drop(-1)
          |> Enum.with_index(1)
          |> Enum.reduce({context, []}, fn {token, index}, {context, steps} ->
            {context, _logits, eval_timings} = timed_eval(context, token)

            step = %{
              index: index,
              token: token,
              piece: token_piece(model, token),
              eval_timings: eval_timings
            }

            {context, [step | steps]}
          end)
        end)

      prompt_eval_steps = Enum.reverse(prompt_eval_steps)

      state = %{
        context: context,
        prompt_tokens: prompt_tokens,
        original_prompt_token_count: length(original_prompt_tokens),
        context_window: context_window,
        prompt_truncated?: length(prompt_tokens) < length(original_prompt_tokens),
        prepared?: profile_prepared?(source_model, opts),
        current_token: seed_token(prompt_tokens),
        prompt_eval_steps: prompt_eval_steps,
        prompt_eval_summary: prompt_eval_summary(prompt_eval_steps)
      }

      {state, [encode_time, prepare_time, prompt_eval_time]}
    end)
  end

  defp profile_backend(%PreparedModel{backend: backend}, _opts), do: backend
  defp profile_backend(_model, opts), do: Map.get(opts, :backend, Llamex.Backend.Nx)

  defp profile_model(%PreparedModel{model: model}), do: model
  defp profile_model(model), do: model

  defp profile_context(%PreparedModel{model: model, backend: backend}, _default_backend) do
    Llamex.Context.new_prepared(model, backend)
  end

  defp profile_context(model, backend), do: Llamex.Context.new(model, backend)

  defp profile_prepared?(%PreparedModel{}, _opts), do: true
  defp profile_prepared?(_model, opts), do: Map.has_key?(opts, :prepared_model)

  defp seed_token([]), do: raise(ArgumentError, "prompt must encode to at least one token")
  defp seed_token(prompt_tokens), do: List.last(prompt_tokens)

  defp sampler(opts) do
    case Map.get(opts, :sampler, :greedy) do
      :greedy -> :greedy
      sampler when is_map(sampler) -> Llamex.Sampler.validate_options!(sampler)
      sampler -> raise ArgumentError, "sampler must be :greedy or a map, got: #{inspect(sampler)}"
    end
  end

  defp timed_step(context, current_token, opts) do
    sampler = sampler(opts)
    history = Map.get(opts, :history, context.tokens)
    sampler_state = Map.get(opts, :sampler_state) || new_sampler_state(sampler)
    candidate_count = Map.get(opts, :candidate_count, 0)

    {context, token, sampler_state, candidates, eval_timings} =
      timed_sample(context, current_token, sampler, sampler_state, history, candidate_count)

    %{
      context: context,
      token: token,
      text: Llamex.decode(context.model, [token]),
      sampler_state: sampler_state,
      candidates: candidates,
      eval_timings: eval_timings
    }
  end

  defp timed_sample(context, current_token, :greedy, sampler_state, history, candidate_count) do
    {context, logits, eval_timings} = timed_eval(context, current_token)
    candidates = candidates(logits, context, :greedy, history, candidate_count)
    token = Llamex.Sampler.greedy(logits, context.backend)

    {context, token, sampler_state, candidates, eval_timings}
  end

  defp timed_sample(
         context,
         current_token,
         %{top_k: top_k} = sampler,
         sampler_state,
         history,
         candidate_count
       )
       when is_integer(top_k) and top_k > 0 do
    {random, sampler_state} = next_random(sampler, sampler_state)

    sampler =
      sampler
      |> Map.put(:random, random)
      |> Map.put(:history, history)

    if fast_top_k_sampling?(context) do
      {context, logits, eval_timings} = timed_eval_top_k(context, current_token, top_k, sampler)
      candidates = candidate_probabilities(logits, sampler, candidate_count)
      token = Llamex.Sampler.sample_candidates(logits, sampler)

      {context, token, sampler_state, candidates, eval_timings}
    else
      {context, logits, eval_timings} = timed_eval(context, current_token)
      candidates = candidates(logits, context, sampler, history, candidate_count)
      token = Llamex.Sampler.sample(logits, context.backend, sampler)

      {context, token, sampler_state, candidates, eval_timings}
    end
  end

  defp timed_sample(context, current_token, sampler, sampler_state, history, candidate_count)
       when is_map(sampler) do
    {random, sampler_state} = next_random(sampler, sampler_state)

    sampler =
      sampler
      |> Map.put(:random, random)
      |> Map.put(:history, history)

    {context, logits, eval_timings} = timed_eval(context, current_token)
    candidates = candidates(logits, context, sampler, history, candidate_count)
    token = Llamex.Sampler.sample(logits, context.backend, sampler)

    {context, token, sampler_state, candidates, eval_timings}
  end

  defp fast_top_k_sampling?(%{
         backend: backend,
         model: %{output: %{weight: weight}}
       })
       when backend in [Llamex.Backend.List, Llamex.Backend.Nx, Llamex.Backend.NxEXLA] and
              not is_nil(weight),
       do: true

  defp fast_top_k_sampling?(_context), do: false

  defp candidates(_logits, _context, _sampler, _history, count) when count <= 0, do: []

  defp candidates(logits, context, :greedy, _history, count) do
    logits
    |> context.backend.to_list()
    |> Enum.with_index()
    |> Enum.sort_by(fn {logit, _token} -> logit end, :desc)
    |> Enum.take(count)
    |> Enum.map(fn {logit, token} -> %{token: token, logit: logit} end)
  end

  defp candidates(logits, context, sampler, history, count) when is_map(sampler) do
    Llamex.Sampler.candidates(
      logits,
      context.backend,
      sampler |> Map.put(:history, history) |> Map.put_new(:random, 0.0),
      count
    )
  end

  defp candidate_probabilities(_logits, _sampler, count) when count <= 0, do: []

  defp candidate_probabilities(logits, sampler, count) do
    Llamex.Sampler.candidate_probabilities(logits, sampler, count)
  end

  defp new_sampler_state(:greedy), do: nil

  defp new_sampler_state(opts) when is_map(opts) do
    seed = Map.get(opts, :seed)

    if seed do
      seed = Llamex.Sampler.validate_seed!(seed)
      :rand.seed_s(:exsss, {seed, seed + 1, seed + 2})
    end
  end

  defp next_random(%{random: random}, sampler_state) when is_float(random),
    do: {Llamex.Sampler.validate_random!(random), sampler_state}

  defp next_random(%{random: _random}, _sampler_state) do
    raise ArgumentError, "random must be a float greater than or equal to zero and less than one"
  end

  defp next_random(_opts, sampler_state) do
    :rand.uniform_s(sampler_state)
  end

  defp timed_eval(%Context{} = context, token) when is_integer(token) and token >= 0 do
    hidden = Map.fetch!(context.model.token_embeddings, token)
    position = context.token_count

    {layer_timings, {context, hidden}} =
      context.model.layers
      |> Enum.with_index()
      |> Enum.reduce({[], {context, hidden}}, fn {layer, layer_index},
                                                 {timings, {context, hidden}} ->
        {layer_time, {context, hidden, component_timings}} =
          timed("layer_#{layer_index}", fn ->
            timed_layer(context, hidden, layer, layer_index, position)
          end)

        step = Map.put(layer_time, :components, component_timings)
        {[step | timings], {context, hidden}}
      end)

    {output_norm_time, hidden} =
      timed("output_norm", fn ->
        maybe_apply_output_norm(
          hidden,
          context.model.output_norm,
          context.model.config.epsilon,
          context.backend
        )
      end)

    {logits_time, logits} =
      timed("logits", fn ->
        timed_logits(context, hidden)
      end)

    eval_timings = %{
      layers: Enum.reverse(layer_timings),
      output_norm: output_norm_time,
      logits: logits_time
    }

    {Context.append(context, token), logits, eval_timings}
  end

  defp timed_eval_top_k(%Context{} = context, token, top_k, opts)
       when is_integer(token) and token >= 0 and is_integer(top_k) and top_k > 0 and is_map(opts) do
    hidden = Map.fetch!(context.model.token_embeddings, token)
    position = context.token_count

    {layer_timings, {context, hidden}} =
      context.model.layers
      |> Enum.with_index()
      |> Enum.reduce({[], {context, hidden}}, fn {layer, layer_index},
                                                 {timings, {context, hidden}} ->
        {layer_time, {context, hidden, component_timings}} =
          timed("layer_#{layer_index}", fn ->
            timed_layer(context, hidden, layer, layer_index, position)
          end)

        step = Map.put(layer_time, :components, component_timings)
        {[step | timings], {context, hidden}}
      end)

    {output_norm_time, hidden} =
      timed("output_norm", fn ->
        maybe_apply_output_norm(
          hidden,
          context.model.output_norm,
          context.model.config.epsilon,
          context.backend
        )
      end)

    {logits_time, logits} =
      timed("top_k_logits", fn ->
        timed_top_k_logits(context, hidden, top_k, opts)
      end)

    eval_timings = %{
      layers: Enum.reverse(layer_timings),
      output_norm: output_norm_time,
      logits: logits_time
    }

    {Context.append(context, token), logits, eval_timings}
  end

  defp timed_layer(context, hidden, layer, layer_index, position) do
    {attention_norm_time, normalized} =
      timed("attention_norm", fn ->
        context.backend.rms_norm(
          hidden,
          Map.fetch!(layer, :attention_norm),
          context.model.config.epsilon
        )
      end)

    {attention_time, {kv_cache, attention}} =
      timed("attention", fn ->
        Attention.forward(
          normalized,
          layer,
          context.kv_cache,
          layer_index,
          position,
          context.model.config.rope_theta,
          context.model.config.rope_dimension_count,
          context.backend
        )
      end)

    hidden = context.backend.add(hidden, attention)

    {mlp_time, {hidden, mlp_timings}} =
      timed("mlp", fn ->
        timed_mlp(hidden, layer, context.model.config.epsilon, context.backend)
      end)

    mlp_time = Map.put(mlp_time, :components, mlp_timings)
    component_timings = [attention_norm_time, attention_time, mlp_time]
    {%{context | kv_cache: kv_cache}, hidden, component_timings}
  end

  defp timed_mlp(
         hidden,
         %{feed_forward_norm: feed_forward_norm} = layer,
         epsilon,
         Llamex.Backend.List
       ) do
    {norm_time, normalized} =
      timed("feed_forward_norm", fn ->
        Llamex.Backend.List.rms_norm(hidden, feed_forward_norm, epsilon)
      end)

    {gate_up_time, {gate, up}} =
      timed("w_gate_up", fn ->
        Llamex.Backend.List.matvec_pair(
          Map.fetch!(layer, :w_gate),
          Map.fetch!(layer, :w_up),
          normalized
        )
      end)

    {activation_time, activated} =
      timed("silu_multiply", fn ->
        gate
        |> Tensor.silu()
        |> Tensor.multiply(up)
      end)

    {down_time, down} =
      timed("w_down", fn ->
        Linear.forward(activated, Map.fetch!(layer, :w_down), Llamex.Backend.List)
      end)

    {residual_time, hidden} =
      timed("residual", fn ->
        Llamex.Backend.List.add(hidden, down)
      end)

    {hidden, [norm_time, gate_up_time, activation_time, down_time, residual_time]}
  end

  defp timed_mlp(hidden, %{feed_forward_norm: feed_forward_norm} = layer, epsilon, backend) do
    {norm_time, normalized} =
      timed("feed_forward_norm", fn ->
        backend.rms_norm(hidden, feed_forward_norm, epsilon)
      end)

    {gate_up_time, {gate, up}} =
      timed("w_gate_up", fn ->
        gate_up_projection(layer, normalized, backend)
      end)

    {activation_time, activated} =
      timed("silu_multiply", fn ->
        gate
        |> backend.silu_multiply(up)
      end)

    {down_time, down} =
      timed("w_down", fn ->
        Map.fetch!(layer, :w_down)
        |> backend.matvec_tensor(activated)
      end)

    {residual_time, hidden} =
      timed("residual", fn ->
        backend.add(hidden, down)
      end)

    {hidden, [norm_time, gate_up_time, activation_time, down_time, residual_time]}
  end

  defp timed_mlp(hidden, _layer, _epsilon, _backend), do: {hidden, []}

  defp maybe_apply_output_norm(hidden, nil, _epsilon, _backend), do: hidden

  defp maybe_apply_output_norm(hidden, output_norm, epsilon, backend) do
    backend.rms_norm(hidden, output_norm, epsilon)
  end

  defp gate_up_projection(
         %{w_gate_up: weight, w_gate_up_row_counts: [gate_count, _up_count]},
         input,
         backend
       ) do
    backend.matvec_split_pair_tensor(weight, gate_count, input)
  end

  defp gate_up_projection(layer, input, backend) do
    backend.matvec_pair_tensor(
      Map.fetch!(layer, :w_gate),
      Map.fetch!(layer, :w_up),
      input
    )
  end

  defp timed_logits(%{model: %{output: %{weight: weight}}, backend: backend}, hidden) do
    backend.matvec_tensor(weight, hidden)
  end

  defp timed_logits(context, hidden) do
    0..(context.model.config.vocab_size - 1)
    |> Enum.map(fn candidate ->
      candidate_embedding = Map.fetch!(context.model.token_embeddings, candidate)

      context.backend.dot(hidden, candidate_embedding)
    end)
    |> context.backend.from_list()
  end

  defp timed_top_k_logits(
         %{model: %{output: %{weight: weight}}, backend: backend},
         hidden,
         top_k,
         opts
       ) do
    backend.top_k_matvec(weight, hidden, top_k,
      history: Map.get(opts, :history, []),
      repetition_penalty: Map.get(opts, :repetition_penalty),
      suppress_tokens: Map.get(opts, :suppress_tokens, [])
    )
  end

  defp token_pieces(model, token_ids) do
    Enum.map(token_ids, &Map.fetch!(model.tokenizer.id_to_token, &1))
  end

  defp token_piece(%{tokenizer: nil}, token_id), do: Integer.to_string(token_id)

  defp token_piece(model, token_id) do
    Map.get(model.tokenizer.id_to_token, token_id, Integer.to_string(token_id))
  end

  defp stop_tokens(opts), do: Llamex.StopTokens.from_options(opts)

  defp stop_token?(token, stop_tokens), do: token in stop_tokens

  defp stop_sequences(opts), do: Llamex.StopSequences.from_options(opts)

  defp stop_sequence?(_text, []), do: false

  defp stop_sequence?(text, stop_sequences) do
    Enum.any?(stop_sequences, &String.contains?(text, &1))
  end

  defp token_info(model, token_id) do
    model.tokenizer
    |> token_type(token_id)
    |> Map.merge(%{
      token: token_id,
      piece: Map.fetch!(model.tokenizer.id_to_token, token_id)
    })
  end

  defp token_candidates(model, candidates) do
    Enum.map(candidates, fn %{token: token} = candidate ->
      model
      |> token_info(token)
      |> Map.merge(Map.delete(candidate, :token))
    end)
  end

  defp token_type(tokenizer, token_id) do
    case Enum.find(tokenizer.token_types, &(&1.id == token_id)) do
      nil -> %{}
      %{type: type, type_id: type_id} -> %{type: type, type_id: type_id}
    end
  end
end
