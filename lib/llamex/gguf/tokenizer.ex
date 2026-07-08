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
    token_types = token_types(metadata, tokens)

    chat_template =
      metadata_value(metadata, "tokenizer.chat_template") ||
        metadata_value(metadata, "tokenizer.ggml.chat_template")

    if merges == [] do
      Llamex.Tokenizer.whitespace(vocab, unknown_token,
        special_tokens: special_tokens,
        token_types: token_types,
        chat_template: chat_template
      )
    else
      Llamex.Tokenizer.bpe(vocab, merges, unknown_token,
        special_tokens: special_tokens,
        token_types: token_types,
        chat_template: chat_template
      )
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

  defp token_types(metadata, tokens) do
    metadata
    |> metadata_array("tokenizer.ggml.token_type", [])
    |> Enum.with_index()
    |> Enum.map(fn {type_id, id} ->
      %{
        id: id,
        token: Enum.at(tokens, id),
        type: token_type_name(type_id),
        type_id: type_id
      }
    end)
  end

  defp token_type_name(1), do: :normal
  defp token_type_name(2), do: :unknown
  defp token_type_name(3), do: :control
  defp token_type_name(4), do: :user_defined
  defp token_type_name(5), do: :unused
  defp token_type_name(6), do: :byte
  defp token_type_name(_type_id), do: :undefined

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
