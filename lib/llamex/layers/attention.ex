defmodule Llamex.Layers.Attention do
  @moduledoc """
  Single-head causal self-attention.
  """

  alias Llamex.{KVCache, Tensor}
  alias Llamex.Layers.Linear

  def forward(input, layer, cache, layer_index)
      when is_list(input) and is_map(layer) and is_integer(layer_index) do
    query = Linear.forward(input, Map.fetch!(layer, :wq))
    key = Linear.forward(input, Map.fetch!(layer, :wk))
    value = Linear.forward(input, Map.fetch!(layer, :wv))
    {cache, entries} = KVCache.append(cache, layer_index, key, value)

    scale = 1.0 / :math.sqrt(length(query))

    weights =
      entries
      |> Enum.map(fn {cached_key, _cached_value} -> Tensor.dot(query, cached_key) * scale end)
      |> Tensor.softmax()

    values = Enum.map(entries, fn {_cached_key, cached_value} -> cached_value end)

    output =
      weights
      |> Tensor.weighted_sum(values)
      |> Linear.forward(Map.fetch!(layer, :wo))

    {cache, output}
  end
end
