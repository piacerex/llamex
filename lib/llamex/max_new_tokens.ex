defmodule Llamex.MaxNewTokens do
  @moduledoc false

  def fetch!(opts) when is_map(opts) do
    opts
    |> Map.fetch!(:max_new_tokens)
    |> validate!()
  end

  def get(opts, default) when is_map(opts) do
    opts
    |> Map.get(:max_new_tokens, default)
    |> validate!()
  end

  def validate!(max_new_tokens) when is_integer(max_new_tokens) and max_new_tokens >= 0 do
    max_new_tokens
  end

  def validate!(max_new_tokens) do
    raise ArgumentError,
          "max_new_tokens must be a non-negative integer, got: #{inspect(max_new_tokens)}"
  end
end
