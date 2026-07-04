defmodule Llamex.Tensor do
  @moduledoc """
  Small list-based tensor helpers.

  These helpers are deliberately plain Elixir. Backend-specific acceleration can
  replace call sites later without changing layer responsibility.
  """

  def add(left, right) when is_list(left) and is_list(right) and length(left) == length(right) do
    left
    |> Enum.zip(right)
    |> Enum.map(fn {a, b} -> a + b end)
  end

  def dot(left, right) when is_list(left) and is_list(right), do: dot(left, right, 0.0)

  defp dot([], [], acc), do: acc

  defp dot([left | left_rest], [right | right_rest], acc) do
    dot(left_rest, right_rest, acc + left * right)
  end

  defp dot(_left, _right, _acc), do: raise(ArgumentError, "vectors must have matching lengths")

  def scale(values, factor) when is_list(values) and is_number(factor) do
    Enum.map(values, &(&1 * factor))
  end

  def multiply(left, right)
      when is_list(left) and is_list(right) and length(left) == length(right) do
    left
    |> Enum.zip(right)
    |> Enum.map(fn {a, b} -> a * b end)
  end

  def silu(values) when is_list(values) do
    Enum.map(values, fn value -> value / (1.0 + :math.exp(-value)) end)
  end

  def softmax(values) when is_list(values) do
    max = Enum.max(values)
    exps = Enum.map(values, &:math.exp(&1 - max))
    total = Enum.sum(exps)

    Enum.map(exps, &(&1 / total))
  end

  def weighted_sum(weights, vectors) when is_list(weights) and is_list(vectors) do
    vectors
    |> Enum.zip(weights)
    |> Enum.reduce(zero_like(List.first(vectors)), fn {vector, weight}, acc ->
      add(acc, scale(vector, weight))
    end)
  end

  def matvec(rows, vector) when is_list(rows) and is_list(vector) do
    Enum.map(rows, &dot(&1, vector))
  end

  def zero_like(values) when is_list(values), do: Enum.map(values, fn _ -> 0.0 end)

  def split_every(values, size) when is_list(values) and is_integer(size) and size > 0 do
    if rem(length(values), size) != 0 do
      raise ArgumentError, "vector length must be divisible by split size"
    end

    Enum.chunk_every(values, size)
  end
end
