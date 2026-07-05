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
    if parallel_matvec?(rows, vector) do
      rows
      |> Enum.chunk_every(matvec_chunk_size())
      |> Task.async_stream(fn chunk -> Enum.map(chunk, &dot(&1, vector)) end,
        ordered: true,
        timeout: :infinity,
        max_concurrency: System.schedulers_online()
      )
      |> Enum.flat_map(fn {:ok, values} -> values end)
    else
      Enum.map(rows, &dot(&1, vector))
    end
  end

  def argmax_matvec(rows, vector) when is_list(rows) and is_list(vector) do
    if parallel_matvec?(rows, vector) do
      rows
      |> Enum.chunk_every(matvec_chunk_size())
      |> Enum.with_index()
      |> Task.async_stream(
        fn {chunk, chunk_index} ->
          chunk
          |> Enum.with_index(chunk_index * matvec_chunk_size())
          |> Enum.reduce(nil, fn {row, index}, best ->
            max_dot(best, index, dot(row, vector))
          end)
        end,
        ordered: false,
        timeout: :infinity,
        max_concurrency: System.schedulers_online()
      )
      |> Enum.reduce(nil, fn {:ok, {index, value}}, best -> max_dot(best, index, value) end)
      |> elem(0)
    else
      rows
      |> Enum.with_index()
      |> Enum.reduce(nil, fn {row, index}, best -> max_dot(best, index, dot(row, vector)) end)
      |> elem(0)
    end
  end

  defp max_dot(nil, index, value), do: {index, value}

  defp max_dot({best_index, best_value}, index, value) do
    if value > best_value, do: {index, value}, else: {best_index, best_value}
  end

  defp parallel_matvec?(rows, vector) do
    length(rows) * length(vector) >= 1_000_000 and System.schedulers_online() > 1
  end

  defp matvec_chunk_size, do: 256

  def zero_like(values) when is_list(values), do: Enum.map(values, fn _ -> 0.0 end)

  def split_every(values, size) when is_list(values) and is_integer(size) and size > 0 do
    if rem(length(values), size) != 0 do
      raise ArgumentError, "vector length must be divisible by split size"
    end

    Enum.chunk_every(values, size)
  end
end
