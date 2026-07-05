defmodule Llamex.Tokenizer.ByteTokens do
  @moduledoc false

  def encode(tokenizer, token) do
    token
    |> :binary.bin_to_list()
    |> Enum.map(&byte_token/1)
    |> Enum.map(&Map.fetch(tokenizer.token_to_id, &1))
    |> collect_ids()
  end

  def decode(tokenizer, tokens, fallback) when is_function(fallback, 2) do
    if has_byte_tokens?(tokenizer, tokens) do
      tokens
      |> Enum.reject(&control_token?(tokenizer, &1))
      |> Enum.map(&decode_token(tokenizer, &1))
      |> decode_pieces()
    else
      fallback.(tokenizer, tokens)
    end
  end

  defp collect_ids(results) do
    if Enum.all?(results, &match?({:ok, _id}, &1)) do
      Enum.map(results, fn {:ok, id} -> id end)
    else
      :error
    end
  end

  defp has_byte_tokens?(tokenizer, tokens) do
    byte_token_ids = byte_token_ids(tokenizer)

    Enum.any?(tokens, &MapSet.member?(byte_token_ids, &1))
  end

  defp control_token?(tokenizer, id) do
    Enum.any?(tokenizer.token_types, &(&1.id == id and &1.type == :control))
  end

  defp decode_pieces(pieces) do
    if Enum.any?(pieces, &String.contains?(&1, "▁")) do
      Llamex.Tokenizer.TextDecoder.decode_pieces(pieces)
    else
      IO.iodata_to_binary(pieces)
    end
  end

  defp decode_token(tokenizer, id) do
    if MapSet.member?(byte_token_ids(tokenizer), id) do
      <<byte_value(Map.fetch!(tokenizer.id_to_token, id))>>
    else
      Map.fetch!(tokenizer.id_to_token, id)
    end
  end

  defp byte_token_ids(tokenizer) do
    tokenizer.token_types
    |> Enum.filter(&(&1.type == :byte))
    |> Enum.map(& &1.id)
    |> MapSet.new()
  end

  defp byte_token(byte), do: "<0x" <> Base.encode16(<<byte>>) <> ">"

  defp byte_value("<0x" <> rest) do
    rest
    |> String.trim_trailing(">")
    |> String.to_integer(16)
  end
end
