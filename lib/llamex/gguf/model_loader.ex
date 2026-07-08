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
    architecture = metadata_value(gguf.metadata, "general.architecture", nil)
    tensor_names = Enum.map(gguf.tensors, & &1.name)
    tensors = tensors_from_reader(gguf, binary, architecture)

    %{
      "config" => config_from_metadata(gguf.metadata),
      "tokenizer" => tokenizer_from_metadata(gguf.metadata),
      "tensor_schema" => Llamex.GGUF.TensorSchema.summary(architecture, tensor_names),
      "tensors" => tensors
    }
  end

  defp tensors_from_reader(%Llamex.GGUF.Reader{} = gguf, binary, architecture) do
    gguf
    |> Llamex.GGUF.Reader.read_tensor_data(binary)
    |> then(&Llamex.GGUF.TensorSchema.normalize_tensor_map(architecture, &1))
  end

  defp config_from_metadata(metadata) do
    prefix = metadata_prefix(metadata)

    %{
      "vocab_size" =>
        metadata_value(metadata, metadata_key(prefix, "vocab_size"), token_count(metadata)),
      "embedding_size" => metadata_value!(metadata, metadata_key(prefix, "embedding_length")),
      "context_size" => metadata_value(metadata, metadata_key(prefix, "context_length"), nil),
      "epsilon" =>
        metadata_value(metadata, metadata_key(prefix, "attention.layer_norm_rms_epsilon"), 1.0e-6),
      "rope_theta" => metadata_value(metadata, metadata_key(prefix, "rope.freq_base"), 10_000.0),
      "rope_dimension_count" =>
        metadata_value(metadata, metadata_key(prefix, "rope.dimension_count"), nil),
      "block_count" => metadata_value(metadata, metadata_key(prefix, "block_count"), nil),
      "attention_head_count" =>
        metadata_value(metadata, metadata_key(prefix, "attention.head_count"), nil),
      "attention_head_count_kv" =>
        metadata_value(metadata, metadata_key(prefix, "attention.head_count_kv"), nil),
      "feed_forward_size" =>
        metadata_value(metadata, metadata_key(prefix, "feed_forward_length"), nil)
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

  defp metadata_prefix(metadata) do
    case metadata_value(metadata, "general.architecture", nil) do
      architecture when is_binary(architecture) ->
        if Map.has_key?(metadata, metadata_key(architecture, "embedding_length")) do
          architecture
        else
          "llama"
        end

      _other ->
        "llama"
    end
  end

  defp metadata_key(prefix, suffix), do: "#{prefix}.#{suffix}"
end
