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

    if max_new_tokens < 0 do
      raise ArgumentError, "max_new_tokens must be zero or positive"
    end

    prompt_tokens = Llamex.encode(model, prompt)
    context = Context.new(model, backend)
    context = ingest_prompt(context, prompt_tokens)

    {context, generated_tokens} =
      generate_tokens(context, seed_token(prompt_tokens), max_new_tokens, stop_token)

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

  defp generate_tokens(context, _current_token, 0, _stop_token), do: {context, []}

  defp generate_tokens(context, current_token, remaining, stop_token) do
    {context, next_token} = Engine.next_token(context, current_token, &Sampler.greedy/2)

    if next_token == stop_token do
      {context, [next_token]}
    else
      {context, rest} = generate_tokens(context, next_token, remaining - 1, stop_token)
      {context, [next_token | rest]}
    end
  end
end
