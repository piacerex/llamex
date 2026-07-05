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
      |> Enum.chunk_every(matvec_chunk_size(rows))
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

  def matvec_pair(left_rows, right_rows, vector)
      when is_list(left_rows) and is_list(right_rows) and is_list(vector) do
    if length(left_rows) != length(right_rows) do
      raise ArgumentError, "matrices must have matching row counts"
    end

    if parallel_matvec?(left_rows, vector) do
      left_rows
      |> Enum.chunk_every(matvec_chunk_size(left_rows))
      |> Enum.zip(Enum.chunk_every(right_rows, matvec_chunk_size(right_rows)))
      |> Task.async_stream(
        fn {left_chunk, right_chunk} ->
          left_values = Enum.map(left_chunk, &dot(&1, vector))
          right_values = Enum.map(right_chunk, &dot(&1, vector))
          {left_values, right_values}
        end,
        ordered: true,
        timeout: :infinity,
        max_concurrency: System.schedulers_online()
      )
      |> Enum.reduce({[], []}, fn {:ok, {left_values, right_values}}, {left_acc, right_acc} ->
        {[left_values | left_acc], [right_values | right_acc]}
      end)
      |> then(fn {left_chunks, right_chunks} ->
        {left_chunks |> Enum.reverse() |> List.flatten(),
         right_chunks |> Enum.reverse() |> List.flatten()}
      end)
    else
      left_values = Enum.map(left_rows, &dot(&1, vector))
      right_values = Enum.map(right_rows, &dot(&1, vector))
      {left_values, right_values}
    end
  end

  def argmax_matvec(rows, vector) when is_list(rows) and is_list(vector) do
    if parallel_matvec?(rows, vector) do
      chunk_size = argmax_matvec_chunk_size(rows)

      rows
      |> Enum.chunk_every(chunk_size)
      |> Enum.with_index()
      |> Task.async_stream(
        fn {chunk, chunk_index} ->
          chunk
          |> Enum.with_index(chunk_index * chunk_size)
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

  def top_k_matvec(rows, vector, top_k, opts \\ [])
      when is_list(rows) and is_list(vector) and is_integer(top_k) and top_k > 0 and
             is_list(opts) do
    history = opts |> Keyword.get(:history, []) |> MapSet.new()
    penalty = Keyword.get(opts, :repetition_penalty)

    if parallel_matvec?(rows, vector) do
      chunk_size = top_k_matvec_chunk_size(rows)

      rows
      |> Enum.chunk_every(chunk_size)
      |> Enum.with_index()
      |> Task.async_stream(
        fn {chunk, chunk_index} ->
          chunk
          |> Enum.with_index(chunk_index * chunk_size)
          |> Enum.reduce([], fn {row, index}, top ->
            value = row |> dot(vector) |> maybe_penalize(index, history, penalty)
            insert_top_k({value, index}, top, top_k)
          end)
        end,
        ordered: false,
        timeout: :infinity,
        max_concurrency: System.schedulers_online()
      )
      |> Enum.reduce([], fn {:ok, chunk_top}, top ->
        Enum.reduce(chunk_top, top, &insert_top_k(&1, &2, top_k))
      end)
      |> Enum.reverse()
    else
      rows
      |> Enum.with_index()
      |> Enum.reduce([], fn {row, index}, top ->
        value = row |> dot(vector) |> maybe_penalize(index, history, penalty)
        insert_top_k({value, index}, top, top_k)
      end)
      |> Enum.reverse()
    end
  end

  defp max_dot(nil, index, value), do: {index, value}

  defp max_dot({best_index, best_value}, index, value) do
    if value > best_value, do: {index, value}, else: {best_index, best_value}
  end

  defp parallel_matvec?(rows, vector) do
    length(rows) * length(vector) >= 1_000_000 and System.schedulers_online() > 1
  end

  defp matvec_chunk_size(_rows), do: 64
  defp argmax_matvec_chunk_size(_rows), do: 256
  defp top_k_matvec_chunk_size(_rows), do: 256

  defp maybe_penalize(value, index, history, penalty)
       when is_number(penalty) and penalty > 0.0 do
    if MapSet.member?(history, index) do
      if value >= 0.0, do: value / penalty, else: value * penalty
    else
      value
    end
  end

  defp maybe_penalize(value, _index, _history, _penalty), do: value

  defp insert_top_k(candidate, [], _top_k), do: [candidate]

  defp insert_top_k({value, _index}, [{lowest, _lowest_index} | _rest] = top, top_k)
       when length(top) == top_k and value <= lowest do
    top
  end

  defp insert_top_k(candidate, top, top_k) do
    [candidate | top]
    |> Enum.sort_by(fn {value, _index} -> value end)
    |> trim_lowest(top_k)
  end

  defp trim_lowest(top, top_k) when length(top) > top_k, do: tl(top)
  defp trim_lowest(top, _top_k), do: top

  def zero_like(values) when is_list(values), do: Enum.map(values, fn _ -> 0.0 end)

  def split_every(values, size) when is_list(values) and is_integer(size) and size > 0 do
    if rem(length(values), size) != 0 do
      raise ArgumentError, "vector length must be divisible by split size"
    end

    Enum.chunk_every(values, size)
  end
end
