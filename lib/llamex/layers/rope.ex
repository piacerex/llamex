defmodule Llamex.Layers.RoPE do
  @moduledoc """
  Rotary positional embedding for query and key vectors.
  """

  def apply(vector, position, theta, dimension_count \\ nil)
      when is_list(vector) and is_integer(position) and position >= 0 and is_number(theta) do
    dimension_count = dimension_count || length(vector) - rem(length(vector), 2)

    if dimension_count == 0 do
      vector
    else
      if dimension_count > length(vector) do
        raise ArgumentError, "RoPE dimension count cannot exceed vector length"
      end

      if rem(dimension_count, 2) != 0 do
        raise ArgumentError, "RoPE vector length must be even"
      end

      {rotary, pass_through} = Enum.split(vector, dimension_count)
      half = div(dimension_count, 2)
      {left, right} = Enum.split(rotary, half)

      {rotated_left, rotated_right} =
        left
        |> Enum.zip(right)
        |> Enum.with_index()
        |> Enum.reduce({[], []}, fn {{x0, x1}, pair_index}, {rotated_left, rotated_right} ->
          angle = position / :math.pow(theta, 2 * pair_index / dimension_count)
          cos = :math.cos(angle)
          sin = :math.sin(angle)

          {[x0 * cos - x1 * sin | rotated_left], [x0 * sin + x1 * cos | rotated_right]}
        end)

      Enum.reverse(rotated_left) ++ Enum.reverse(rotated_right) ++ pass_through
    end
  end
end
