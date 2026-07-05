defmodule Llamex.Profile do
  @moduledoc """
  Small profiling helpers for local GGUF generation experiments.
  """

  alias Llamex.{Context, Tensor}
  alias Llamex.Layers.{Attention, Linear, RMSNorm, SwiGLU}

  def timed(label, fun) when is_binary(label) and is_function(fun, 0) do
    {microseconds, result} = :timer.tc(fun)

    {%{label: label, milliseconds: div(microseconds, 1000)}, result}
  end

  def generation_step(model, prompt, opts) when is_binary(prompt) and is_map(opts) do
    backend = Map.get(opts, :backend, Llamex.Backend.List)
    sampler = Map.get(opts, :sampler, :greedy)

    {prefill_time, {state, prefill_timings}} = timed_prefill(model, prompt, backend)

    {step_time, step} =
      timed("step", fn ->
        timed_step(state.context, state.current_token, %{sampler: sampler})
      end)

    %{
      prompt_tokens: length(state.prompt_tokens),
      token: step.token,
      text: step.text,
      eval_timings: step.eval_timings,
      prefill_timings: prefill_timings,
      timings: [prefill_time, step_time]
    }
  end

  def prefill_steps(model, prompt, opts) when is_binary(prompt) and is_map(opts) do
    backend = Map.get(opts, :backend, Llamex.Backend.List)
    prompt_tokens = Llamex.encode(model, prompt)
    context = Llamex.Context.new(model, backend)
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
          piece: Map.fetch!(model.tokenizer.id_to_token, token),
          timing: timing
        }

        {[step | steps], context}
      end)

    %{
      prompt_tokens: prompt_tokens,
      current_token: List.last(prompt_tokens),
      current_piece: Map.fetch!(model.tokenizer.id_to_token, List.last(prompt_tokens)),
      context_tokens: context.tokens,
      steps: Enum.reverse(steps)
    }
  end

  def generation_steps(model, prompt, opts) when is_binary(prompt) and is_map(opts) do
    backend = Map.get(opts, :backend, Llamex.Backend.List)
    sampler = Map.get(opts, :sampler, :greedy)
    max_new_tokens = Map.get(opts, :max_new_tokens, 1)
    stop_tokens = stop_tokens(opts)

    {prefill_time, {state, prefill_timings}} = timed_prefill(model, prompt, backend)

    {steps, _context, _current_token, _sampler_state, finish_reason} =
      Enum.reduce_while(
        1..max_new_tokens,
        {[], state.context, state.current_token, nil, :length},
        fn index, {steps, context, current_token, sampler_state, _finish_reason} ->
          {step_time, step} =
            timed("step_#{index}", fn ->
              timed_step(context, current_token, %{
                sampler: sampler,
                sampler_state: sampler_state,
                history: context.tokens
              })
            end)

          step_info =
            model
            |> token_info(step.token)
            |> Map.merge(%{
              index: index,
              text: step.text,
              timing: step_time,
              eval_timings: step.eval_timings
            })

          finish_reason = if stop_token?(step.token, stop_tokens), do: :stop, else: :length

          next_state =
            {[step_info | steps], step.context, step.token, step.sampler_state, finish_reason}

          if stop_token?(step.token, stop_tokens) do
            {:halt, next_state}
          else
            {:cont, next_state}
          end
        end
      )

    steps = Enum.reverse(steps)
    generated_tokens = Enum.map(steps, & &1.token)

    %{
      backend: backend,
      max_new_tokens: max_new_tokens,
      stop_token: List.first(stop_tokens),
      stop_tokens: stop_tokens,
      sampler: sampler,
      prompt_tokens: length(state.prompt_tokens),
      prompt_token_ids: state.prompt_tokens,
      prompt_pieces: token_pieces(model, state.prompt_tokens),
      generated_tokens: generated_tokens,
      generated_pieces: token_pieces(model, generated_tokens),
      generated_token_info: Enum.map(generated_tokens, &token_info(model, &1)),
      finish_reason: finish_reason,
      text: Llamex.decode(model, generated_tokens),
      prefill_timings: prefill_timings,
      timings: [prefill_time | Enum.map(steps, & &1.timing)],
      steps: steps
    }
  end

  defp timed_prefill(model, prompt, backend) do
    timed("prefill", fn ->
      {encode_time, prompt_tokens} =
        timed("prompt_encode", fn ->
          Llamex.encode(model, prompt)
        end)

      {prepare_time, context} =
        timed("backend_prepare", fn ->
          Llamex.Context.new(model, backend)
        end)

      {prompt_eval_time, context} =
        timed("prompt_eval", fn ->
          prompt_tokens
          |> Enum.drop(-1)
          |> Enum.reduce(context, fn token, context ->
            {context, _logits, _eval_timings} = timed_eval(context, token)
            context
          end)
        end)

      state = %{
        context: context,
        prompt_tokens: prompt_tokens,
        current_token: seed_token(prompt_tokens)
      }

      {state, [encode_time, prepare_time, prompt_eval_time]}
    end)
  end

  defp seed_token([]), do: raise(ArgumentError, "prompt must encode to at least one token")
  defp seed_token(prompt_tokens), do: List.last(prompt_tokens)

  defp timed_step(context, current_token, opts) do
    sampler = Map.get(opts, :sampler, :greedy)
    history = Map.get(opts, :history, context.tokens)
    sampler_state = Map.get(opts, :sampler_state) || new_sampler_state(sampler)

    {context, logits, eval_timings} = timed_eval(context, current_token)

    {token, sampler_state} =
      case sampler do
        :greedy ->
          {Llamex.Sampler.greedy(logits, context.backend), sampler_state}

        sampler when is_map(sampler) ->
          {random, sampler_state} = next_random(sampler, sampler_state)

          token =
            logits
            |> Llamex.Sampler.sample(
              context.backend,
              sampler |> Map.put(:random, random) |> Map.put(:history, history)
            )

          {token, sampler_state}
      end

    %{
      context: context,
      token: token,
      text: Llamex.decode(context.model, [token]),
      sampler_state: sampler_state,
      eval_timings: eval_timings
    }
  end

  defp new_sampler_state(:greedy), do: nil

  defp new_sampler_state(opts) when is_map(opts) do
    seed = Map.get(opts, :seed)

    if seed do
      :rand.seed_s(:exsss, {seed, seed + 1, seed + 2})
    end
  end

  defp next_random(%{random: random}, sampler_state) when is_float(random),
    do: {random, sampler_state}

  defp next_random(_opts, sampler_state) do
    :rand.uniform_s(sampler_state)
  end

  defp timed_eval(%Context{} = context, token) when is_integer(token) and token >= 0 do
    hidden = Map.fetch!(context.model.token_embeddings, token)
    position = length(context.tokens)

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
        maybe_apply_output_norm(hidden, context.model.output_norm, context.model.config.epsilon)
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

  defp timed_layer(context, hidden, layer, layer_index, position) do
    {attention_norm_time, normalized} =
      timed("attention_norm", fn ->
        RMSNorm.forward(hidden, Map.fetch!(layer, :attention_norm), context.model.config.epsilon)
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
          context.backend
        )
      end)

    hidden = Tensor.add(hidden, attention)

    {mlp_time, hidden} =
      timed("mlp", fn ->
        maybe_apply_mlp(hidden, layer, context.model.config.epsilon, context.backend)
      end)

    component_timings = [attention_norm_time, attention_time, mlp_time]
    {%{context | kv_cache: kv_cache}, hidden, component_timings}
  end

  defp maybe_apply_mlp(hidden, %{feed_forward_norm: feed_forward_norm} = layer, epsilon, backend) do
    feed_forward =
      hidden
      |> RMSNorm.forward(feed_forward_norm, epsilon)
      |> SwiGLU.forward(layer, backend)

    Tensor.add(hidden, feed_forward)
  end

  defp maybe_apply_mlp(hidden, _layer, _epsilon, _backend), do: hidden

  defp maybe_apply_output_norm(hidden, nil, _epsilon), do: hidden

  defp maybe_apply_output_norm(hidden, output_norm, epsilon) do
    RMSNorm.forward(hidden, output_norm, epsilon)
  end

  defp timed_logits(%{model: %{output: %{weight: weight}}, backend: backend}, hidden) do
    hidden
    |> Linear.forward(weight, backend)
    |> backend.from_list()
  end

  defp timed_logits(context, hidden) do
    0..(context.model.config.vocab_size - 1)
    |> Enum.map(fn candidate ->
      candidate_embedding = Map.fetch!(context.model.token_embeddings, candidate)

      Tensor.dot(hidden, candidate_embedding)
    end)
    |> context.backend.from_list()
  end

  defp token_pieces(model, token_ids) do
    Enum.map(token_ids, &Map.fetch!(model.tokenizer.id_to_token, &1))
  end

  defp stop_tokens(%{stop_tokens: stop_tokens}) when is_list(stop_tokens), do: stop_tokens
  defp stop_tokens(%{stop_token: nil}), do: []
  defp stop_tokens(%{stop_token: stop_token}) when is_integer(stop_token), do: [stop_token]
  defp stop_tokens(_opts), do: []

  defp stop_token?(token, stop_tokens), do: token in stop_tokens

  defp token_info(model, token_id) do
    model.tokenizer
    |> token_type(token_id)
    |> Map.merge(%{
      token: token_id,
      piece: Map.fetch!(model.tokenizer.id_to_token, token_id)
    })
  end

  defp token_type(tokenizer, token_id) do
    case Enum.find(tokenizer.token_types, &(&1.id == token_id)) do
      nil -> %{}
      %{type: type, type_id: type_id} -> %{type: type, type_id: type_id}
    end
  end
end
