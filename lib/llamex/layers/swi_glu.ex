defmodule Llamex.Layers.SwiGLU do
  @moduledoc """
  Llama-style gated feed-forward layer.
  """

  def forward(input, layer, backend \\ Llamex.Backend.List)

  def forward(input, layer, backend)
      when is_map(layer) do
    {gate, up} = gate_up_projection(layer, input, backend)

    activated = backend.silu_multiply(gate, up)

    layer
    |> Map.fetch!(:w_down)
    |> backend.matvec_tensor(activated)
  end

  defp gate_up_projection(
         %{w_gate_up: weight, w_gate_up_row_counts: [gate_count, _up_count]},
         input,
         backend
       ) do
    backend.matvec_split_pair_tensor(weight, gate_count, input)
  end

  defp gate_up_projection(layer, input, backend) do
    backend.matvec_pair_tensor(
      Map.fetch!(layer, :w_gate),
      Map.fetch!(layer, :w_up),
      input
    )
  end
end
