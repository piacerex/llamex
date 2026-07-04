defmodule Llamex.Layers.RMSNorm do
  @moduledoc """
  Root-mean-square normalization.
  """

  def forward(input, weight, epsilon)
      when is_list(input) and is_list(weight) and length(input) == length(weight) do
    mean_square =
      input
      |> Enum.reduce(0.0, fn value, acc -> acc + value * value end)
      |> Kernel./(length(input))

    scale = 1.0 / :math.sqrt(mean_square + epsilon)

    input
    |> Enum.zip(weight)
    |> Enum.map(fn {value, gain} -> value * scale * gain end)
  end
end
