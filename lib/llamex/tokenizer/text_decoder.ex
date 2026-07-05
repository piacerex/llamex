defmodule Llamex.Tokenizer.TextDecoder do
  @moduledoc false

  def decode(tokenizer, tokens) when is_list(tokens) do
    pieces =
      tokens
      |> Enum.reject(&control_token?(tokenizer, &1))
      |> Enum.map(&Map.fetch!(tokenizer.id_to_token, &1))

    decode_pieces(pieces)
  end

  def decode_pieces(pieces) when is_list(pieces) do
    if Enum.any?(pieces, &String.contains?(&1, "▁")) do
      pieces
      |> Enum.join("")
      |> String.replace("▁", " ")
      |> String.trim_leading()
    else
      Enum.join(pieces, " ")
    end
  end

  defp control_token?(tokenizer, id) do
    Enum.any?(tokenizer.token_types, &(&1.id == id and &1.type == :control))
  end
end
