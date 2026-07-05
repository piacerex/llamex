defmodule Llamex.Generation do
  @moduledoc """
  Prompt-to-text generation loop.
  """

  alias Llamex.{Context, Engine, Model, Sampler}

  def prefill(%Model{} = model, prompt, opts) when is_binary(prompt) and is_map(opts) do
    backend = Map.fetch!(opts, :backend)
    prompt_tokens = Llamex.encode(model, prompt)
    context = Context.new(model, backend)
    context = ingest_prompt(context, prompt_tokens)

    %{
      context: context,
      prompt_tokens: prompt_tokens,
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
    {context, logits} = Engine.eval(context, current_token)
    {next_token, sampler_state} = sample(logits, context.backend, sampler, sampler_state, history)

    {context, next_token, sampler_state}
  end

  def generate(%Model{} = model, prompt, opts)
      when is_binary(prompt) and is_map(opts) do
    backend = Map.fetch!(opts, :backend)
    max_new_tokens = Map.fetch!(opts, :max_new_tokens)
    stop_tokens = stop_tokens(opts)
    sampler = Map.get(opts, :sampler, :greedy)

    if max_new_tokens < 0 do
      raise ArgumentError, "max_new_tokens must be zero or positive"
    end

    %{context: context, prompt_tokens: prompt_tokens, current_token: current_token} =
      prefill(model, prompt, %{backend: backend})

    sampler_state = new_sampler_state(sampler)

    {context, generated_tokens, finish_reason} =
      generate_tokens(
        context,
        current_token,
        max_new_tokens,
        stop_tokens,
        sampler,
        sampler_state,
        prompt_tokens,
        []
      )

    %{
      text: Llamex.decode(model, generated_tokens),
      prompt_tokens: prompt_tokens,
      generated_tokens: generated_tokens,
      finish_reason: finish_reason,
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
         sampler,
         sampler_state,
         prompt_tokens,
         generated_tokens
       ) do
    {context, logits} = Engine.eval(context, current_token)

    {next_token, sampler_state} =
      sample(
        logits,
        context.backend,
        sampler,
        sampler_state,
        prompt_tokens ++ Enum.reverse(generated_tokens)
      )

    if stop_token?(next_token, stop_tokens) do
      {context, Enum.reverse([next_token | generated_tokens]), :stop}
    else
      generate_tokens(
        context,
        next_token,
        remaining - 1,
        stop_tokens,
        sampler,
        sampler_state,
        prompt_tokens,
        [next_token | generated_tokens]
      )
    end
  end

  defp stop_tokens(%{stop_tokens: stop_tokens}) when is_list(stop_tokens), do: stop_tokens
  defp stop_tokens(%{stop_token: nil}), do: []
  defp stop_tokens(%{stop_token: stop_token}) when is_integer(stop_token), do: [stop_token]
  defp stop_tokens(_opts), do: []

  defp stop_token?(token, stop_tokens), do: token in stop_tokens

  defp new_sampler_state(:greedy), do: nil

  defp new_sampler_state(opts) when is_map(opts) do
    seed = Map.get(opts, :seed)

    if seed do
      :rand.seed_s(:exsss, {seed, seed + 1, seed + 2})
    end
  end

  defp sample(logits, backend, :greedy, sampler_state, _history) do
    {Sampler.greedy(logits, backend), sampler_state}
  end

  defp sample(logits, backend, opts, sampler_state, history) when is_map(opts) do
    {random, sampler_state} = next_random(opts, sampler_state)

    token =
      logits
      |> Sampler.sample(backend, opts |> Map.put(:random, random) |> Map.put(:history, history))

    {token, sampler_state}
  end

  defp next_random(%{random: random}, sampler_state) when is_float(random),
    do: {random, sampler_state}

  defp next_random(_opts, sampler_state) do
    :rand.uniform_s(sampler_state)
  end
end
