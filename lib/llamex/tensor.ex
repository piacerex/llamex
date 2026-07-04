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

  def dot(left, right) when is_list(left) and is_list(right) and length(left) == length(right) do
    left
    |> Enum.zip(right)
    |> Enum.reduce(0.0, fn {a, b}, acc -> acc + a * b end)
  end

  def scale(values, factor) when is_list(values) and is_number(factor) do
    Enum.map(values, &(&1 * factor))
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
end
