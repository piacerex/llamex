defmodule Llamex.Layers.Attention do
  @moduledoc """
  Causal self-attention.
  """

  alias Llamex.{KVCache, Tensor}

  def forward(
        input,
        layer,
        cache,
        layer_index,
        position,
        rope_theta,
        rope_dimension_count \\ nil,
        backend \\ Llamex.Backend.List
      )
      when is_map(layer) and is_integer(layer_index) do
    head_count = Map.get(layer, :head_count, 1)
    kv_head_count = Map.get(layer, :kv_head_count, head_count)

    {query_heads, key_heads, value_heads} =
      qkv_heads(
        layer,
        input,
        head_count,
        kv_head_count,
        position,
        rope_theta,
        rope_dimension_count,
        backend
      )

    {cache, entries} = KVCache.append(cache, layer_index, key_heads, value_heads)

    {cache, entries} =
      KVCache.prepare_entries(cache, layer_index, backend, entries, key_heads, value_heads)

    output =
      query_heads
      |> backend.attend_heads(entries, head_count, kv_head_count)
      |> then(&backend.matvec_tensor(Map.fetch!(layer, :wo), &1))

    {cache, output}
  end

  defp qkv_projection(
         %{w_qkv: weight, w_qkv_row_counts: [q_count, k_count, v_count]},
         input,
         backend
       ) do
    backend.matvec_split_triple(weight, q_count, k_count, v_count, input)
  end

  defp qkv_projection(layer, input, backend) do
    backend.matvec_triple(
      Map.fetch!(layer, :wq),
      Map.fetch!(layer, :wk),
      Map.fetch!(layer, :wv),
      input
    )
  end

  defp qkv_heads(
         %{w_qkv: weight, w_qkv_row_counts: counts},
         input,
         head_count,
         kv_head_count,
         position,
         rope_theta,
         rope_dimension_count,
         backend
       ) do
    backend.qkv_heads(
      weight,
      counts,
      input,
      head_count,
      kv_head_count,
      position,
      rope_theta,
      rope_dimension_count
    )
  end

  defp qkv_heads(
         layer,
         input,
         head_count,
         kv_head_count,
         position,
         rope_theta,
         rope_dimension_count,
         backend
       ) do
    {query, key, value} = qkv_projection(layer, input, backend)

    query_heads =
      query
      |> split_heads(head_count)
      |> apply_rope(position, rope_theta, rope_dimension_count, backend)

    key_heads =
      key
      |> split_heads(kv_head_count)
      |> apply_rope(position, rope_theta, rope_dimension_count, backend)

    {query_heads, key_heads, split_heads(value, kv_head_count)}
  end

  defp split_heads(vector, head_count) do
    Tensor.split_every(vector, div(length(vector), head_count))
  end

  defp apply_rope(heads, position, rope_theta, rope_dimension_count, backend) do
    Enum.map(heads, &backend.rope(&1, position, rope_theta, rope_dimension_count))
  end
end
