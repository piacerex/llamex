defmodule Llamex.GGUF.Tokenizer do
  @moduledoc """
  Builds Llamex tokenizers from GGUF metadata.
  """

  def from_metadata(metadata) when is_map(metadata) do
    tokens = metadata_array!(metadata, "tokenizer.ggml.tokens")
    vocab = tokens |> Enum.with_index() |> Map.new()
    unknown_token = unknown_token(metadata, tokens)
    merges = metadata_array(metadata, "tokenizer.ggml.merges", [])
    special_tokens = special_tokens(metadata, tokens)

    if merges == [] do
      Llamex.Tokenizer.whitespace(vocab, unknown_token, special_tokens: special_tokens)
    else
      Llamex.Tokenizer.bpe(vocab, merges, unknown_token, special_tokens: special_tokens)
    end
  end

  defp unknown_token(metadata, tokens) do
    case metadata_value(metadata, "tokenizer.ggml.unknown_token_id") do
      nil -> List.first(tokens)
      id when is_integer(id) -> Enum.at(tokens, id)
    end
  end

  defp special_tokens(metadata, tokens) do
    %{}
    |> put_special_token(metadata, tokens, :unknown, "tokenizer.ggml.unknown_token_id")
    |> put_special_token(metadata, tokens, :bos, "tokenizer.ggml.bos_token_id")
    |> put_special_token(metadata, tokens, :eos, "tokenizer.ggml.eos_token_id")
    |> put_special_token(metadata, tokens, :padding, "tokenizer.ggml.padding_token_id")
    |> put_special_flag(metadata, :add_bos, "tokenizer.ggml.add_bos_token")
    |> put_special_flag(metadata, :add_eos, "tokenizer.ggml.add_eos_token")
  end

  defp put_special_token(attrs, metadata, tokens, name, key) do
    case metadata_value(metadata, key) do
      id when is_integer(id) ->
        Map.put(attrs, name, %{id: id, token: Enum.at(tokens, id)})

      _other ->
        attrs
    end
  end

  defp put_special_flag(attrs, metadata, name, key) do
    case metadata_value(metadata, key) do
      value when is_boolean(value) -> Map.put(attrs, name, value)
      _other -> attrs
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
