defmodule Llamex.GGUF.TensorSchema do
  @moduledoc """
  Architecture-specific GGUF tensor schema names used by diagnostics.
  """

  @token_embedding_name "token_embd.weight"

  @llama_interesting_names [
    @token_embedding_name,
    "output_norm.weight",
    "output.weight",
    "blk.0.attn_norm.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight",
    "blk.0.attn_v.weight",
    "blk.0.attn_output.weight",
    "blk.0.ffn_norm.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight"
  ]

  @gemma3_interesting_names [
    @token_embedding_name,
    "output_norm.weight",
    "output.weight",
    "blk.0.attn_norm.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight",
    "blk.0.attn_v.weight",
    "blk.0.attn_output.weight",
    "blk.0.attn_q_norm.weight",
    "blk.0.attn_k_norm.weight",
    "blk.0.post_attention_norm.weight",
    "blk.0.post_ffw_norm.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight"
  ]

  def required_tensor_names(_architecture), do: [@token_embedding_name]

  def token_embedding_name(_architecture), do: @token_embedding_name

  def interesting_tensor_names("gemma3"), do: @gemma3_interesting_names
  def interesting_tensor_names(_architecture), do: @llama_interesting_names

  def normalize_tensor_map(architecture, tensors) when is_map(tensors) do
    Map.new(tensors, fn {name, tensor} -> {normalize_name(architecture, name), tensor} end)
  end

  def mappings(architecture, tensor_names) when is_list(tensor_names) do
    tensor_names
    |> Enum.map(fn name -> {name, normalize_name(architecture, name)} end)
    |> Enum.reject(fn {name, schema_name} -> name == schema_name end)
    |> Enum.map(fn {name, schema_name} -> %{name: name, schema_name: schema_name} end)
  end

  def unmapped_names(architecture, tensor_names) when is_list(tensor_names) do
    tensor_names
    |> Enum.reject(fn name -> recognized_name?(architecture, name) end)
  end

  def unsupported_feature_names(architecture, tensor_names) when is_list(tensor_names) do
    tensor_names
    |> Enum.filter(fn name -> unsupported_feature_name?(architecture, name) end)
  end

  def normalize_name("gemma3", "blk." <> rest = name) do
    case String.split(rest, ".", parts: 3) do
      [index, "post_attention_norm", suffix] -> "blk.#{index}.ffn_norm.#{suffix}"
      _other -> name
    end
  end

  def normalize_name(_architecture, name), do: name

  def recognized_name?(architecture, name) when is_binary(name) do
    internal_name?(normalize_name(architecture, name)) or
      unsupported_feature_name?(architecture, name)
  end

  def unsupported_feature_name?("gemma3", "blk." <> rest) do
    case String.split(rest, ".", parts: 3) do
      [_index, part, "weight"] -> part in ["attn_q_norm", "attn_k_norm", "post_ffw_norm"]
      _other -> false
    end
  end

  def unsupported_feature_name?(_architecture, _name), do: false

  def schema_shape([columns, rows]), do: [rows, columns]
  def schema_shape(dimensions), do: dimensions

  defp internal_name?(name)
       when name in [@token_embedding_name, "output_norm.weight", "output.weight"] do
    true
  end

  defp internal_name?("blk." <> rest) do
    case String.split(rest, ".", parts: 3) do
      [_index, part, "weight"] ->
        part in [
          "attn_norm",
          "attn_q",
          "attn_k",
          "attn_v",
          "attn_output",
          "ffn_norm",
          "ffn_gate",
          "ffn_up",
          "ffn_down"
        ]

      _other ->
        false
    end
  end

  defp internal_name?(_name), do: false
end
