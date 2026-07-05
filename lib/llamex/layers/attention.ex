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
        backend \\ Llamex.Backend.List
      )
      when is_list(input) and is_map(layer) and is_integer(layer_index) do
    head_count = Map.get(layer, :head_count, 1)
    kv_head_count = Map.get(layer, :kv_head_count, head_count)

    query_heads =
      input
      |> Linear.forward(Map.fetch!(layer, :wq), backend)
      |> RoPE.apply(position, rope_theta)
      |> split_heads(head_count)

    key_heads =
      input
      |> Linear.forward(Map.fetch!(layer, :wk), backend)
      |> RoPE.apply(position, rope_theta)
      |> split_heads(kv_head_count)

    value_heads =
      input
      |> Linear.forward(Map.fetch!(layer, :wv), backend)
      |> split_heads(kv_head_count)

    {cache, entries} = KVCache.append(cache, layer_index, key_heads, value_heads)

    output =
      query_heads
      |> Enum.with_index()
      |> Enum.flat_map(fn {query, head_index} ->
        attend_head(query, entries, kv_head_index(head_index, head_count, kv_head_count))
      end)
      |> Linear.forward(Map.fetch!(layer, :wo), backend)

    {cache, output}
  end

  defp kv_head_index(head_index, head_count, kv_head_count) do
    div(head_index * kv_head_count, head_count)
  end

  defp split_heads(vector, head_count) do
    Tensor.split_every(vector, div(length(vector), head_count))
  end

  defp attend_head(query, entries, head_index) do
    scale = 1.0 / :math.sqrt(length(query))

    weights =
      entries
      |> Enum.map(fn {cached_keys, _cached_values} ->
        cached_key = Enum.at(cached_keys, head_index)
        Tensor.dot(query, cached_key) * scale
      end)
      |> Tensor.softmax()

    values =
      Enum.map(entries, fn {_cached_keys, cached_values} ->
        Enum.at(cached_values, head_index)
      end)

    Tensor.weighted_sum(weights, values)
  end
end
