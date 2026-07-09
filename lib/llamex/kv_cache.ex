defmodule Llamex.KVCache do
  @moduledoc """
  Key/value cache for autoregressive attention.
  """

  @enforce_keys [:layers, :prepared_layers]
  defstruct [:layers, :prepared_layers]

  @type t :: %__MODULE__{
          layers: %{optional(non_neg_integer()) => list({term(), term()})},
          prepared_layers: %{optional({non_neg_integer(), module()}) => term()}
        }

  def new do
    %__MODULE__{layers: %{}, prepared_layers: %{}}
  end

  def append(%__MODULE__{} = cache, layer_index, key, value)
      when is_integer(layer_index) and layer_index >= 0 do
    layers =
      Map.update(cache.layers, layer_index, [{key, value}], fn entries ->
        [{key, value} | entries]
      end)

    {%{cache | layers: layers}, Map.fetch!(layers, layer_index)}
  end

  def append_window(%__MODULE__{} = cache, layer_index, key, value, window)
      when is_integer(layer_index) and layer_index >= 0 and is_integer(window) and window > 0 do
    {cache, entries} = append(cache, layer_index, key, value)
    pruned_entries = Enum.take(entries, window)

    if length(pruned_entries) == length(entries) do
      {cache, entries}
    else
      layers = Map.put(cache.layers, layer_index, pruned_entries)
      prepared_layers = drop_prepared_layer(cache.prepared_layers, layer_index)
      {%{cache | layers: layers, prepared_layers: prepared_layers}, pruned_entries}
    end
  end

  def prepare_entries(%__MODULE__{} = cache, layer_index, backend, entries, key, value)
      when is_integer(layer_index) and layer_index >= 0 and is_atom(backend) and is_list(entries) do
    prepared_key = {layer_index, backend}

    prepared =
      case Map.fetch(cache.prepared_layers, prepared_key) do
        {:ok, prepared} -> backend.append_kv_entry(prepared, key, value)
        :error -> backend.prepare_kv_entries(entries)
      end

    cache = %{cache | prepared_layers: Map.put(cache.prepared_layers, prepared_key, prepared)}

    {cache, prepared}
  end

  def entries(%__MODULE__{} = cache, layer_index)
      when is_integer(layer_index) and layer_index >= 0 do
    Map.get(cache.layers, layer_index, [])
    |> Enum.reverse()
  end

  def entry_counts(%__MODULE__{} = cache) do
    Map.new(cache.layers, fn {layer_index, entries} ->
      {layer_index, length(entries)}
    end)
  end

  def entry_count(%__MODULE__{} = cache) do
    cache.layers
    |> Map.values()
    |> Enum.map(&length/1)
    |> Enum.sum()
  end

  defp drop_prepared_layer(prepared_layers, layer_index) do
    Map.reject(prepared_layers, fn
      {{^layer_index, _backend}, _prepared} -> true
      {_key, _prepared} -> false
    end)
  end
end
