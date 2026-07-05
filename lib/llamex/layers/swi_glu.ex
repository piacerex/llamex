defmodule Llamex.Layers.SwiGLU do
  @moduledoc """
  Llama-style gated feed-forward layer.
  """

  alias Llamex.Tensor
  alias Llamex.Layers.Linear

  def forward(input, layer, backend \\ Llamex.Backend.List)

  def forward(input, layer, backend)
      when is_list(input) and is_map(layer) do
    {gate, up} =
      backend.matvec_pair(
        Map.fetch!(layer, :w_gate),
        Map.fetch!(layer, :w_up),
        input
      )

    gate
    |> Tensor.silu()
    |> Tensor.multiply(up)
    |> Linear.forward(Map.fetch!(layer, :w_down), backend)
  end
end
