defmodule Llamex.KVCache do
  @moduledoc """
  Key/value cache for autoregressive attention.
  """

  @enforce_keys [:layers]
  defstruct [:layers]

  @type t :: %__MODULE__{
          layers: %{optional(non_neg_integer()) => list({list(number()), list(number())})}
        }

  def new do
    %__MODULE__{layers: %{}}
  end

  def append(%__MODULE__{} = cache, layer_index, key, value)
      when is_integer(layer_index) and layer_index >= 0 and is_list(key) and is_list(value) do
    layers =
      Map.update(cache.layers, layer_index, [{key, value}], fn entries ->
        [{key, value} | entries]
      end)

    {%{cache | layers: layers}, Map.fetch!(layers, layer_index)}
  end

  def entries(%__MODULE__{} = cache, layer_index)
      when is_integer(layer_index) and layer_index >= 0 do
    Map.get(cache.layers, layer_index, [])
    |> Enum.reverse()
  end
end
