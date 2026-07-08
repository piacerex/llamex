defmodule Llamex.ContextWindow do
  @moduledoc """
  Prompt token windowing helpers.
  """

  def resolve(%{config: %{context_size: context_size}}, opts) when is_map(opts) do
    opts
    |> Map.get(:context_window, context_size)
    |> validate()
  end

  def apply(tokens, nil) when is_list(tokens), do: tokens

  def apply(tokens, context_window) when is_list(tokens) do
    context_window = validate(context_window)
    Enum.take(tokens, -context_window)
  end

  def validate(nil), do: nil

  def validate(context_window) when is_integer(context_window) and context_window > 0 do
    context_window
  end

  def validate(context_window) do
    raise ArgumentError,
          "context_window must be a positive integer, got: #{inspect(context_window)}"
  end

  def generation_budget(max_new_tokens, _prompt_token_count, nil)
      when is_integer(max_new_tokens) do
    max_new_tokens
  end

  def generation_budget(max_new_tokens, prompt_token_count, context_window)
      when is_integer(max_new_tokens) and is_integer(prompt_token_count) and
             is_integer(context_window) do
    max(0, min(max_new_tokens, context_window - prompt_token_count + 1))
  end

  def context_limited?(requested_max_new_tokens, effective_max_new_tokens)
      when is_integer(requested_max_new_tokens) and is_integer(effective_max_new_tokens) do
    effective_max_new_tokens < requested_max_new_tokens
  end
end
