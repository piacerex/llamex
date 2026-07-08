defmodule Llamex.GGUF.ModelConfig do
  @moduledoc """
  Converts GGUF model metadata into Llamex model config maps.
  """

  @fields [
    {"vocab_size", "vocab_size"},
    {"embedding_size", "embedding_length"},
    {"context_size", "context_length"},
    {"epsilon", "attention.layer_norm_rms_epsilon"},
    {"rope_theta", "rope.freq_base"},
    {"rope_dimension_count", "rope.dimension_count"},
    {"block_count", "block_count"},
    {"attention_head_count", "attention.head_count"},
    {"attention_head_count_kv", "attention.head_count_kv"},
    {"feed_forward_size", "feed_forward_length"}
  ]

  def fields, do: @fields

  def surface(architectures) when is_list(architectures) do
    Map.new(architectures, fn architecture ->
      {
        architecture,
        Enum.map(@fields, fn {name, suffix} ->
          %{name: name, metadata_key: metadata_key(architecture, suffix)}
        end)
      }
    end)
  end

  def from_metadata(metadata) when is_map(metadata) do
    prefix = metadata_prefix(metadata)

    %{
      "vocab_size" =>
        metadata_value(metadata, metadata_key(prefix, "vocab_size"), token_count!(metadata)),
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

  def partial_from_metadata(metadata) when is_map(metadata) do
    prefix = metadata_prefix(metadata)

    %{
      "vocab_size" =>
        metadata_value(metadata, metadata_key(prefix, "vocab_size"), token_count(metadata)),
      "embedding_size" => metadata_value(metadata, metadata_key(prefix, "embedding_length"), nil),
      "context_size" => metadata_value(metadata, metadata_key(prefix, "context_length"), nil),
      "epsilon" =>
        metadata_value(metadata, metadata_key(prefix, "attention.layer_norm_rms_epsilon"), nil),
      "rope_theta" => metadata_value(metadata, metadata_key(prefix, "rope.freq_base"), nil),
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

  def metadata_prefix(metadata) when is_map(metadata) do
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

  defp token_count(metadata) do
    case metadata_value(metadata, "tokenizer.ggml.tokens", nil) do
      %{values: values} -> length(values)
      nil -> nil
    end
  end

  defp token_count!(metadata) do
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

  defp metadata_key(prefix, suffix), do: "#{prefix}.#{suffix}"
end
