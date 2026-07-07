defmodule Llamex.Generation do
  @moduledoc """
  Prompt-to-text generation loop.
  """

  alias Llamex.{Context, ContextWindow, Engine, Model, PreparedModel, Sampler}

  def prefill(%Model{} = model, prompt, opts) when is_binary(prompt) and is_map(opts) do
    backend = Map.fetch!(opts, :backend)
    original_prompt_tokens = Llamex.encode(model, prompt)
    context_window = ContextWindow.resolve(model, opts)
    prompt_tokens = ContextWindow.apply(original_prompt_tokens, context_window)
    context = Context.new(model, backend)
    context = ingest_prompt(context, prompt_tokens)

    %{
      context: context,
      prompt_tokens: prompt_tokens,
      original_prompt_token_count: length(original_prompt_tokens),
      context_window: context_window,
      prompt_truncated?: length(prompt_tokens) < length(original_prompt_tokens),
      current_token: seed_token(prompt_tokens)
    }
  end

  def prefill(%PreparedModel{} = prepared_model, prompt, opts)
      when is_binary(prompt) and is_map(opts) do
    model = prepared_model.model
    original_prompt_tokens = Llamex.encode(model, prompt)
    context_window = ContextWindow.resolve(model, opts)
    prompt_tokens = ContextWindow.apply(original_prompt_tokens, context_window)
    context = Context.new_prepared(model, prepared_model.backend)
    context = ingest_prompt(context, prompt_tokens)

    %{
      context: context,
      prompt_tokens: prompt_tokens,
      original_prompt_token_count: length(original_prompt_tokens),
      context_window: context_window,
      prompt_truncated?: length(prompt_tokens) < length(original_prompt_tokens),
      current_token: seed_token(prompt_tokens)
    }
  end

  def step(%Context{} = context, current_token, opts)
      when is_integer(current_token) and current_token >= 0 and is_map(opts) do
    sampler = Map.get(opts, :sampler, :greedy)
    {context, next_token, sampler_state} = step_token(context, current_token, sampler, opts)

    %{
      context: context,
      token: next_token,
      text: Llamex.decode(context.model, [next_token]),
      sampler_state: sampler_state
    }
  end

  defp step_token(context, current_token, :greedy, _opts) do
    {context, next_token} = Engine.greedy_next_token(context, current_token)
    {context, next_token, nil}
  end

  defp step_token(context, current_token, sampler, opts) do
    history = Map.get(opts, :history, context.tokens)
    sampler_state = Map.get(opts, :sampler_state) || new_sampler_state(sampler)

    {next_token, context, sampler_state} =
      sample_next(context, current_token, sampler, sampler_state, history)

    {context, next_token, sampler_state}
  end

  defp sample_next(context, current_token, :greedy, sampler_state, _history) do
    {context, next_token} = Engine.greedy_next_token(context, current_token)
    {next_token, context, sampler_state}
  end

  defp sample_next(context, current_token, %{top_k: top_k} = sampler, sampler_state, history)
       when is_integer(top_k) and top_k > 0 do
    {random, sampler_state} = next_random(sampler, sampler_state)

    opts =
      sampler
      |> Map.put(:random, random)
      |> Map.put(:history, history)
      |> put_dynamic_suppressions(context)

    if fast_top_k_sampling?(context) do
      {next_context, candidates} = Engine.eval_top_k(context, current_token, top_k, opts)

      if candidates == [] do
        {context, logits} = Engine.eval(context, current_token)
        {Sampler.sample(logits, context.backend, opts), context, sampler_state}
      else
        {Sampler.sample_candidates(candidates, Map.delete(opts, :suppress_tokens)), next_context,
         sampler_state}
      end
    else
      {context, logits} = Engine.eval(context, current_token)
      {Sampler.sample(logits, context.backend, opts), context, sampler_state}
    end
  end

  defp sample_next(context, current_token, sampler, sampler_state, history) do
    {random, sampler_state} = next_random(sampler, sampler_state)
    {context, logits} = Engine.eval(context, current_token)

    token =
      logits
      |> Sampler.sample(
        context.backend,
        sampler
        |> Map.put(:random, random)
        |> Map.put(:history, history)
        |> put_dynamic_suppressions(context)
      )

    {token, context, sampler_state}
  end

  defp fast_top_k_sampling?(%{
         backend: backend,
         model: %{output: %{weight: weight}}
       })
       when backend in [Llamex.Backend.List, Llamex.Backend.Nx, Llamex.Backend.NxEXLA] and
              not is_nil(weight),
       do: true

  defp fast_top_k_sampling?(_context), do: false

  defp put_dynamic_suppressions(opts, context) do
    opts
    |> put_no_repeat_ngram_suppressions()
    |> put_no_repeat_adjacent_word_suppressions(context)
  end

  defp put_no_repeat_ngram_suppressions(%{no_repeat_ngram_size: size, history: history} = opts)
       when is_integer(size) and size > 1 and is_list(history) do
    suppressed = no_repeat_ngram_tokens(history, size)

    if suppressed == [] do
      opts
    else
      Map.update(opts, :suppress_tokens, suppressed, &Enum.uniq(&1 ++ suppressed))
    end
  end

  defp put_no_repeat_ngram_suppressions(opts), do: opts

  defp put_no_repeat_adjacent_word_suppressions(
         %{no_repeat_adjacent_word: true, history: history} = opts,
         context
       )
       when is_list(history) and history != [] do
    suppressed = repeated_word_tokens(context.model, List.last(history))

    if suppressed == [] do
      opts
    else
      Map.update(opts, :suppress_tokens, suppressed, &Enum.uniq(&1 ++ suppressed))
    end
  end

  defp put_no_repeat_adjacent_word_suppressions(opts, _context), do: opts

  defp no_repeat_ngram_tokens(history, size) when length(history) < size - 1, do: []

  defp no_repeat_ngram_tokens(history, size) do
    prefix = Enum.take(history, -(size - 1))

    history
    |> Enum.chunk_every(size, 1, :discard)
    |> Enum.flat_map(fn ngram ->
      if Enum.take(ngram, size - 1) == prefix do
        [List.last(ngram)]
      else
        []
      end
    end)
    |> Enum.uniq()
  end

  defp repeated_word_tokens(%{tokenizer: nil}, _last_token), do: []
  defp repeated_word_tokens(model, _last_token) when not is_map_key(model, :tokenizer), do: []

  defp repeated_word_tokens(model, last_token) do
    with word when is_binary(word) <- token_word(model, last_token) do
      model.tokenizer.id_to_token
      |> Enum.flat_map(fn {token, piece} ->
        if piece_word(piece) == word, do: [token], else: []
      end)
    else
      _ -> []
    end
  end

  defp token_word(model, token) do
    model.tokenizer.id_to_token
    |> Map.get(token)
    |> piece_word()
  end

  defp piece_word(nil), do: nil

  defp piece_word(piece) do
    piece
    |> String.trim_leading("▁")
    |> then(&Regex.run(~r/[[:alnum:]]+/u, &1))
    |> case do
      [word] -> String.downcase(word)
      _ -> nil
    end
  end

  def generate(%Model{} = model, prompt, opts)
      when is_binary(prompt) and is_map(opts) do
    max_new_tokens = Map.fetch!(opts, :max_new_tokens)
    stop_tokens = stop_tokens(opts)
    stop_sequences = stop_sequences(opts)
    sampler = Map.get(opts, :sampler, :greedy)

    if max_new_tokens < 0 do
      raise ArgumentError, "max_new_tokens must be zero or positive"
    end

    state = prefill(model, prompt, Map.take(opts, [:backend, :context_window]))
    %{context: context, prompt_tokens: prompt_tokens, current_token: current_token} = state

    effective_max_new_tokens =
      ContextWindow.generation_budget(max_new_tokens, length(prompt_tokens), state.context_window)

    sampler_state = new_sampler_state(sampler)

    {context, generated_tokens, finish_reason} =
      generate_tokens(
        context,
        current_token,
        effective_max_new_tokens,
        stop_tokens,
        stop_sequences,
        sampler,
        sampler_state,
        prompt_tokens,
        []
      )

    %{
      text: Llamex.decode(model, generated_tokens),
      prompt_tokens: prompt_tokens,
      original_prompt_token_count: state.original_prompt_token_count,
      context_window: state.context_window,
      prompt_truncated?: state.prompt_truncated?,
      exla: exla_info(context.backend),
      requested_max_new_tokens: max_new_tokens,
      effective_max_new_tokens: effective_max_new_tokens,
      generated_tokens: generated_tokens,
      finish_reason: finish_reason(finish_reason, max_new_tokens, effective_max_new_tokens),
      context: context
    }
  end

  def generate(%PreparedModel{} = prepared_model, prompt, opts)
      when is_binary(prompt) and is_map(opts) do
    generate_prepared(prepared_model, prompt, opts)
  end

  defp generate_prepared(%PreparedModel{} = prepared_model, prompt, opts) do
    model = prepared_model.model
    max_new_tokens = Map.fetch!(opts, :max_new_tokens)
    stop_tokens = stop_tokens(opts)
    stop_sequences = stop_sequences(opts)
    sampler = Map.get(opts, :sampler, :greedy)

    if max_new_tokens < 0 do
      raise ArgumentError, "max_new_tokens must be zero or positive"
    end

    state = prefill(prepared_model, prompt, Map.take(opts, [:context_window]))
    %{context: context, prompt_tokens: prompt_tokens, current_token: current_token} = state

    effective_max_new_tokens =
      ContextWindow.generation_budget(max_new_tokens, length(prompt_tokens), state.context_window)

    sampler_state = new_sampler_state(sampler)

    {context, generated_tokens, finish_reason} =
      generate_tokens(
        context,
        current_token,
        effective_max_new_tokens,
        stop_tokens,
        stop_sequences,
        sampler,
        sampler_state,
        prompt_tokens,
        []
      )

    %{
      text: Llamex.decode(model, generated_tokens),
      prompt_tokens: prompt_tokens,
      original_prompt_token_count: state.original_prompt_token_count,
      context_window: state.context_window,
      prompt_truncated?: state.prompt_truncated?,
      exla: exla_info(context.backend),
      requested_max_new_tokens: max_new_tokens,
      effective_max_new_tokens: effective_max_new_tokens,
      generated_tokens: generated_tokens,
      finish_reason: finish_reason(finish_reason, max_new_tokens, effective_max_new_tokens),
      context: context
    }
  end

  defp ingest_prompt(context, []), do: context
  defp ingest_prompt(context, [_token]), do: context

  defp ingest_prompt(context, prompt_tokens) do
    prompt_tokens
    |> Enum.drop(-1)
    |> Enum.reduce(context, fn token, context ->
      {context, _logits} = Engine.eval(context, token)
      context
    end)
  end

  defp seed_token([]), do: raise(ArgumentError, "prompt must encode to at least one token")
  defp seed_token(prompt_tokens), do: List.last(prompt_tokens)

  defp generate_tokens(
         context,
         _current_token,
         0,
         _stop_tokens,
         _stop_sequences,
         _sampler,
         _sampler_state,
         _prompt_tokens,
         generated_tokens
       ) do
    {context, Enum.reverse(generated_tokens), :length}
  end

  defp generate_tokens(
         context,
         current_token,
         remaining,
         stop_tokens,
         stop_sequences,
         sampler,
         sampler_state,
         prompt_tokens,
         generated_tokens
       ) do
    history = prompt_tokens ++ Enum.reverse(generated_tokens)

    {next_token, context, sampler_state} =
      sample_next(context, current_token, sampler, sampler_state, history)

    generated_tokens = [next_token | generated_tokens]
    generated_text = Llamex.decode(context.model, Enum.reverse(generated_tokens))

    cond do
      stop_token?(next_token, stop_tokens) ->
        {context, Enum.reverse(generated_tokens), :stop}

      stop_sequence?(generated_text, stop_sequences) ->
        {context, Enum.reverse(generated_tokens), :stop_sequence}

      true ->
        generate_tokens(
          context,
          next_token,
          remaining - 1,
          stop_tokens,
          stop_sequences,
          sampler,
          sampler_state,
          prompt_tokens,
          generated_tokens
        )
    end
  end

  defp stop_tokens(%{stop_tokens: stop_tokens}) when is_list(stop_tokens), do: stop_tokens
  defp stop_tokens(%{stop_token: nil}), do: []
  defp stop_tokens(%{stop_token: stop_token}) when is_integer(stop_token), do: [stop_token]
  defp stop_tokens(_opts), do: []

  defp stop_token?(token, stop_tokens), do: token in stop_tokens

  defp stop_sequences(%{stop_sequences: stop_sequences}) when is_list(stop_sequences) do
    Enum.filter(stop_sequences, &(is_binary(&1) and &1 != ""))
  end

  defp stop_sequences(%{stop_sequence: stop_sequence}) when is_binary(stop_sequence) do
    if stop_sequence == "", do: [], else: [stop_sequence]
  end

  defp stop_sequences(_opts), do: []

  defp stop_sequence?(_text, []), do: false

  defp stop_sequence?(text, stop_sequences) do
    Enum.any?(stop_sequences, &String.contains?(text, &1))
  end

  defp exla_info(backend) when backend in [Llamex.Backend.Nx, Llamex.Backend.NxEXLA] do
    Llamex.Backend.NxEXLA.configured()
  end

  defp exla_info(_backend), do: nil

  defp finish_reason(:length, requested_max_new_tokens, effective_max_new_tokens) do
    if ContextWindow.context_limited?(requested_max_new_tokens, effective_max_new_tokens) do
      :context_window
    else
      :length
    end
  end

  defp finish_reason(reason, _requested_max_new_tokens, _effective_max_new_tokens), do: reason

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
end
