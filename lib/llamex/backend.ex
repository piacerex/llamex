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
  @callback matvec(tensor(), tensor()) :: list(number())
  @callback matvec_tensor(tensor(), tensor()) :: tensor()
  @callback top_k_matvec(tensor(), tensor(), pos_integer(), keyword()) ::
              list({number(), non_neg_integer()})
  @callback rope(tensor(), non_neg_integer(), number(), pos_integer() | nil) :: list(number())
  @callback matvec_pair(tensor(), tensor(), tensor()) :: {list(number()), list(number())}
  @callback matvec_pair_tensor(tensor(), tensor(), tensor()) :: {tensor(), tensor()}
  @callback matvec_split_pair_tensor(tensor(), pos_integer(), tensor()) :: {tensor(), tensor()}
  @callback matvec_triple(tensor(), tensor(), tensor(), tensor()) ::
              {list(number()), list(number()), list(number())}
  @callback matvec_split_triple(tensor(), pos_integer(), pos_integer(), pos_integer(), tensor()) ::
              {list(number()), list(number()), list(number())}
  @callback silu_multiply(tensor(), tensor()) :: tensor()
  @callback rms_norm(tensor(), tensor(), number()) :: tensor()
  @callback attend_head(tensor(), tensor(), tensor()) :: list(number())
  @callback attend_heads(
              list(tensor()),
              list({list(tensor()), list(tensor())}),
              pos_integer(),
              pos_integer()
            ) ::
              list(number())
  @callback add(tensor(), tensor()) :: tensor()
  @callback argmax(tensor()) :: non_neg_integer()
  @callback to_list(tensor()) :: list(number())
end
