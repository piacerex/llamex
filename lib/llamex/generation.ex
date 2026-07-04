defmodule Llamex.Generation do
  @moduledoc """
  Prompt-to-text generation loop.
  """

  alias Llamex.{Context, Engine, Model, Sampler}

  def generate(%Model{} = model, prompt, opts)
      when is_binary(prompt) and is_map(opts) do
    backend = Map.fetch!(opts, :backend)
    max_new_tokens = Map.fetch!(opts, :max_new_tokens)
    stop_token = Map.get(opts, :stop_token)
    sampler = Map.get(opts, :sampler, :greedy)

    if max_new_tokens < 0 do
      raise ArgumentError, "max_new_tokens must be zero or positive"
    end

    prompt_tokens = Llamex.encode(model, prompt)
    context = Context.new(model, backend)
    context = ingest_prompt(context, prompt_tokens)
    sampler_state = new_sampler_state(sampler)

    {context, generated_tokens} =
      generate_tokens(
        context,
        seed_token(prompt_tokens),
        max_new_tokens,
        stop_token,
        sampler,
        sampler_state,
        prompt_tokens,
        []
      )

    %{
      text: Llamex.decode(model, generated_tokens),
      prompt_tokens: prompt_tokens,
      generated_tokens: generated_tokens,
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
         _stop_token,
         _sampler,
         _sampler_state,
         _prompt_tokens,
         generated_tokens
       ) do
    {context, Enum.reverse(generated_tokens)}
  end

  defp generate_tokens(
         context,
         current_token,
         remaining,
         stop_token,
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

    if next_token == stop_token do
      {context, Enum.reverse([next_token | generated_tokens])}
    else
      generate_tokens(
        context,
        next_token,
        remaining - 1,
        stop_token,
        sampler,
        sampler_state,
        prompt_tokens,
        [next_token | generated_tokens]
      )
    end
  end

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
