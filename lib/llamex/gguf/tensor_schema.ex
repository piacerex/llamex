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
    "blk.0.post_attention_norm.weight",
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

  def normalize_name("gemma3", "blk." <> rest = name) do
    case String.split(rest, ".", parts: 3) do
      [index, "post_attention_norm", suffix] -> "blk.#{index}.ffn_norm.#{suffix}"
      _other -> name
    end
  end

  def normalize_name(_architecture, name), do: name

  def schema_shape([columns, rows]), do: [rows, columns]
  def schema_shape(dimensions), do: dimensions
end
