defmodule Llamex.StopTokens do
  @moduledoc false

  def from_options(%{stop_tokens: stop_tokens}) when is_list(stop_tokens) do
    Enum.map(stop_tokens, &normalize_token/1)
  end

  def from_options(%{stop_tokens: stop_tokens}) do
    raise ArgumentError,
          "stop_tokens must be a list of non-negative integers, got: #{inspect(stop_tokens)}"
  end

  def from_options(%{stop_token: nil}), do: []

  def from_options(%{stop_token: stop_token}) do
    [normalize_token(stop_token)]
  end

  def from_options(_opts), do: []

  defp normalize_token(token) when is_integer(token) and token >= 0, do: token

  defp normalize_token(token) do
    raise ArgumentError, "stop token must be a non-negative integer, got: #{inspect(token)}"
  end
end
