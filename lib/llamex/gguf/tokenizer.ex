defmodule Llamex.GGUF.Tokenizer do
  @moduledoc """
  Builds Llamex tokenizers from GGUF metadata.
  """

  def from_metadata(metadata) when is_map(metadata) do
    tokens = metadata_array!(metadata, "tokenizer.ggml.tokens")
    vocab = tokens |> Enum.with_index() |> Map.new()
    unknown_token = unknown_token(metadata, tokens)
    merges = metadata_array(metadata, "tokenizer.ggml.merges", [])

    if merges == [] do
      Llamex.Tokenizer.whitespace(vocab, unknown_token)
    else
      Llamex.Tokenizer.bpe(vocab, merges, unknown_token)
    end
  end

  defp unknown_token(metadata, tokens) do
    case metadata_value(metadata, "tokenizer.ggml.unknown_token_id") do
      nil -> List.first(tokens)
      id when is_integer(id) -> Enum.at(tokens, id)
    end
  end

  defp metadata_array!(metadata, key) do
    case metadata_array(metadata, key, nil) do
      nil -> raise ArgumentError, "GGUF metadata missing #{key}"
      values -> values
    end
  end

  defp metadata_array(metadata, key, default) do
    case metadata_value(metadata, key) do
      %{values: values} -> values
      nil -> default
    end
  end

  defp metadata_value(metadata, key) do
    case Map.fetch(metadata, key) do
      {:ok, %{value: value}} -> value
      :error -> nil
    end
  end
end
