defmodule Llamex.Backend do
  @moduledoc """
  Tensor operation boundary used by the inference engine.

  Backends keep the engine small: AtomVM-oriented code can use the list backend,
  while BEAM builds that have Nx available can opt into the Nx backend.
  """

  @type tensor :: term()

  @callback from_list(list(number())) :: tensor()
  @callback prepare_model(Llamex.Model.t()) :: Llamex.Model.t()
  @callback dot(tensor(), tensor()) :: number()
  @callback matvec(tensor(), list(number())) :: list(number())
  @callback matvec_pair(tensor(), tensor(), list(number())) :: {list(number()), list(number())}
  @callback add(tensor(), tensor()) :: tensor()
  @callback argmax(tensor()) :: non_neg_integer()
  @callback to_list(tensor()) :: list(number())
end
