defmodule Llamex.Layers.SwiGLU do
  @moduledoc """
  Llama-style gated feed-forward layer.
  """

  alias Llamex.Tensor
  alias Llamex.Layers.Linear

  def forward(input, layer) when is_list(input) and is_map(layer) do
    gate =
      input
      |> Linear.forward(Map.fetch!(layer, :w_gate))
      |> Tensor.silu()

    up = Linear.forward(input, Map.fetch!(layer, :w_up))

    gate
    |> Tensor.multiply(up)
    |> Linear.forward(Map.fetch!(layer, :w_down))
  end
end
