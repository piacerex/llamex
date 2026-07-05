defmodule Llamex.Layers.Linear do
  @moduledoc """
  Dense projection layer.
  """

  alias Llamex.Tensor

  def forward(input, weights) when is_list(input) and is_list(weights) do
    Tensor.matvec(weights, input)
  end

  def forward(input, weights, backend) when is_list(input) and is_atom(backend) do
    backend.matvec(weights, input)
  end

  def forward(input, weights, bias) when is_list(input) and is_list(weights) and is_list(bias) do
    input
    |> forward(weights)
    |> Tensor.add(bias)
  end
end
