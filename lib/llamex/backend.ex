defmodule Llamex.Backend do
  @moduledoc """
  Tensor operation boundary used by the inference engine.

  Backends keep the engine small: AtomVM-oriented code can use the list backend,
  while BEAM builds that have Nx available can opt into the Nx backend.
  """

  @type tensor :: term()

  @callback from_list(list(number())) :: tensor()
  @callback dot(tensor(), tensor()) :: number()
  @callback add(tensor(), tensor()) :: tensor()
  @callback argmax(tensor()) :: non_neg_integer()
  @callback to_list(tensor()) :: list(number())
end
