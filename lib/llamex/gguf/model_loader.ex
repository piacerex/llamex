defmodule Llamex.GGUF.ModelLoader do
  @moduledoc """
  Loads Llamex models from GGUF files.

  F32, F16, BF16, and dequantized Q2_K/Q3_K/Q4_0/Q4_1/Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0/Q8_1/Q8_K tensor data are supported at this stage.
  """

  def load(path) when is_binary(path) do
    binary = File.read!(path)
    gguf = Llamex.GGUF.Reader.read_binary(binary)
    validate_loadable!(gguf)

    gguf
    |> to_model_map(binary)
    |> Llamex.ModelLoader.from_map()
  end

  def to_model_map(%Llamex.GGUF.Reader{} = gguf, binary) when is_binary(binary) do
    %{
      "config" => config_from_metadata(gguf.metadata),
      "tokenizer" => tokenizer_from_metadata(gguf.metadata),
      "tensors" => Llamex.GGUF.Reader.read_tensor_data(gguf, binary)
    }
  end

  defp config_from_metadata(metadata) do
    %{
      "vocab_size" => metadata_value(metadata, "llama.vocab_size", token_count(metadata)),
      "embedding_size" => metadata_value!(metadata, "llama.embedding_length"),
      "context_size" => metadata_value(metadata, "llama.context_length", nil),
      "epsilon" => metadata_value(metadata, "llama.attention.layer_norm_rms_epsilon", 1.0e-6),
      "rope_theta" => metadata_value(metadata, "llama.rope.freq_base", 10_000.0),
      "rope_dimension_count" => metadata_value(metadata, "llama.rope.dimension_count", nil),
      "block_count" => metadata_value(metadata, "llama.block_count", nil),
      "attention_head_count" => metadata_value(metadata, "llama.attention.head_count", nil),
      "attention_head_count_kv" => metadata_value(metadata, "llama.attention.head_count_kv", nil),
      "feed_forward_size" => metadata_value(metadata, "llama.feed_forward_length", nil)
    }
    |> Enum.reject(fn {_key, value} -> is_nil(value) end)
    |> Map.new()
  end

  defp tokenizer_from_metadata(metadata) do
    tokenizer = Llamex.GGUF.Tokenizer.from_metadata(metadata)

    %{
      "type" => tokenizer_type(tokenizer),
      "unknown_token" => tokenizer.unknown_token,
      "vocab" => tokenizer.token_to_id,
      "special_tokens" => tokenizer.special_tokens,
      "token_types" => tokenizer.token_types,
      "chat_template" => tokenizer.chat_template
    }
    |> put_merges(tokenizer)
  end

  defp tokenizer_type(%Llamex.Tokenizer.BPE{}), do: "bpe"
  defp tokenizer_type(%Llamex.Tokenizer.Whitespace{}), do: "whitespace"

  defp put_merges(attrs, %Llamex.Tokenizer.BPE{merges: merges}) do
    Map.put(attrs, "merges", Enum.map(merges, fn {left, right} -> [left, right] end))
  end

  defp put_merges(attrs, _tokenizer), do: attrs

  defp validate_loadable!(gguf) do
    if Llamex.GGUF.Diagnostic.loadable?(gguf) do
      :ok
    else
      issues =
        gguf
        |> Llamex.GGUF.Diagnostic.compatibility_issues()
        |> Enum.join("; ")

      raise ArgumentError, "GGUF model is not loadable by Llamex: #{issues}"
    end
  end

  defp token_count(metadata) do
    metadata
    |> metadata_value!("tokenizer.ggml.tokens")
    |> Map.fetch!(:values)
    |> length()
  end

  defp metadata_value!(metadata, key) do
    case metadata_value(metadata, key, nil) do
      nil -> raise ArgumentError, "GGUF metadata missing #{key}"
      value -> value
    end
  end

  defp metadata_value(metadata, key, default) do
    case Map.fetch(metadata, key) do
      {:ok, %{value: value}} -> value
      :error -> default
    end
  end
end
