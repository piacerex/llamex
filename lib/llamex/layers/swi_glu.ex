defmodule Llamex.Layers.SwiGLU do
  @moduledoc """
  Llama-style gated feed-forward layer.
  """

  def forward(input, layer, backend \\ Llamex.Backend.List)

  def forward(input, layer, backend)
      when is_list(input) and is_map(layer) do
    {gate, up} =
      backend.matvec_pair_tensor(
        Map.fetch!(layer, :w_gate),
        Map.fetch!(layer, :w_up),
        input
      )

    activated = backend.silu_multiply(gate, up)

    layer
    |> Map.fetch!(:w_down)
    |> backend.matvec_tensor(activated)
    |> backend.to_list()
  end
end
