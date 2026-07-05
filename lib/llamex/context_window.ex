defmodule Llamex.ContextWindow do
  @moduledoc """
  Prompt token windowing helpers.
  """

  def resolve(%{config: %{context_size: context_size}}, opts) when is_map(opts) do
    Map.get(opts, :context_window) || context_size
  end

  def apply(tokens, nil) when is_list(tokens), do: tokens

  def apply(tokens, context_window) when is_list(tokens) and is_integer(context_window) do
    if context_window <= 0 do
      raise ArgumentError, "context_window must be positive"
    end

    Enum.take(tokens, -context_window)
  end
end
