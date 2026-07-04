defmodule Llamex.Layers.RoPE do
  @moduledoc """
  Rotary positional embedding for query and key vectors.
  """

  def apply(vector, position, theta)
      when is_list(vector) and is_integer(position) and position >= 0 and is_number(theta) do
    if rem(length(vector), 2) != 0 do
      raise ArgumentError, "RoPE vector length must be even"
    end

    vector
    |> Enum.chunk_every(2)
    |> Enum.with_index()
    |> Enum.flat_map(fn {[x0, x1], pair_index} ->
      angle = position / :math.pow(theta, 2 * pair_index / length(vector))
      cos = :math.cos(angle)
      sin = :math.sin(angle)

      [x0 * cos - x1 * sin, x0 * sin + x1 * cos]
    end)
  end
end
