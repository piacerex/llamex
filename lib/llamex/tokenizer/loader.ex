defmodule Llamex.Tokenizer.Loader do
  @moduledoc """
  Loads tokenizer files.
  """

  def load_tokenizer_json(path) when is_binary(path) do
    path
    |> File.read!()
    |> JSON.decode!()
    |> from_tokenizer_json()
  end

  def from_tokenizer_json(%{"model" => %{"type" => "BPE"} = model} = attrs) do
    Llamex.Tokenizer.bpe(
      Map.fetch!(model, "vocab"),
      Map.fetch!(model, "merges"),
      unknown_token(attrs)
    )
  end

  def from_tokenizer_json(%{"model" => %{"type" => type}}) do
    raise ArgumentError, "unsupported tokenizer.json model type: #{type}"
  end

  defp unknown_token(%{"model" => %{"unk_token" => token}}) when is_binary(token), do: token
  defp unknown_token(%{"unk_token" => token}) when is_binary(token), do: token

  defp unknown_token(%{"added_tokens" => added_tokens}) when is_list(added_tokens) do
    added_tokens
    |> Enum.find_value(fn
      %{"content" => token, "special" => true} -> token
      _other -> nil
    end)
    |> case do
      nil -> raise ArgumentError, "tokenizer.json must define unk_token"
      token -> token
    end
  end

  defp unknown_token(_attrs), do: raise(ArgumentError, "tokenizer.json must define unk_token")
end
