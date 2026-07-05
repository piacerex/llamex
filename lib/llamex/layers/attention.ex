defmodule Llamex.Layers.Attention do
  @moduledoc """
  Causal self-attention.
  """

  alias Llamex.{KVCache, Tensor}
  alias Llamex.Layers.{Linear, RoPE}

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

    {query, key, value} = qkv_projection(layer, input, backend)

    query_heads =
      query
      |> split_heads(head_count)
      |> apply_rope(position, rope_theta, rope_dimension_count)

    key_heads =
      key
      |> split_heads(kv_head_count)
      |> apply_rope(position, rope_theta, rope_dimension_count)

    value_heads =
      split_heads(value, kv_head_count)

    {cache, entries} = KVCache.append(cache, layer_index, key_heads, value_heads)

    output =
      query_heads
      |> Enum.with_index()
      |> Enum.flat_map(fn {query, head_index} ->
        attend_head(
          query,
          entries,
          kv_head_index(head_index, head_count, kv_head_count),
          backend
        )
      end)
      |> Linear.forward(Map.fetch!(layer, :wo), backend)

    {cache, output}
  end

  defp kv_head_index(head_index, head_count, kv_head_count) do
    div(head_index * kv_head_count, head_count)
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

  defp split_heads(vector, head_count) do
    Tensor.split_every(vector, div(length(vector), head_count))
  end

  defp apply_rope(heads, position, rope_theta, rope_dimension_count) do
    Enum.map(heads, &RoPE.apply(&1, position, rope_theta, rope_dimension_count))
  end

  defp attend_head(query, entries, head_index, backend) do
    keys =
      Enum.map(entries, fn {cached_keys, _cached_values} ->
        Enum.at(cached_keys, head_index)
      end)

    values =
      Enum.map(entries, fn {_cached_keys, cached_values} ->
        Enum.at(cached_values, head_index)
      end)

    backend.attend_head(query, keys, values)
  end
end
