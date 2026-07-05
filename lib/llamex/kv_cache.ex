defmodule Llamex.KVCache do
  @moduledoc """
  Key/value cache for autoregressive attention.
  """

  @enforce_keys [:layers, :prepared_layers]
  defstruct [:layers, :prepared_layers]

  @type t :: %__MODULE__{
          layers: %{optional(non_neg_integer()) => list({list(number()), list(number())})},
          prepared_layers: %{optional({non_neg_integer(), module()}) => term()}
        }

  def new do
    %__MODULE__{layers: %{}, prepared_layers: %{}}
  end

  def append(%__MODULE__{} = cache, layer_index, key, value)
      when is_integer(layer_index) and layer_index >= 0 and is_list(key) and is_list(value) do
    layers =
      Map.update(cache.layers, layer_index, [{key, value}], fn entries ->
        [{key, value} | entries]
      end)

    {%{cache | layers: layers}, Map.fetch!(layers, layer_index)}
  end

  def prepare_entries(%__MODULE__{} = cache, layer_index, backend, entries, key, value)
      when is_integer(layer_index) and layer_index >= 0 and is_atom(backend) and is_list(entries) and
             is_list(key) and is_list(value) do
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
end
