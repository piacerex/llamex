defmodule Llamex.Backend.List do
  @moduledoc """
  Pure Elixir tensor backend.

  This backend intentionally uses plain lists so the core engine remains easy to
  port to restricted runtimes such as AtomVM.
  """

  @behaviour Llamex.Backend

  @impl true
  def from_list(values) when is_list(values), do: values

  @impl true
  def prepare_model(model), do: model

  @impl true
  def dot(left, right) when is_list(left) and is_list(right), do: dot(left, right, 0.0)

  defp dot([], [], acc), do: acc

  defp dot([left | left_rest], [right | right_rest], acc) do
    dot(left_rest, right_rest, acc + left * right)
  end

  defp dot(_left, _right, _acc), do: raise(ArgumentError, "vectors must have matching lengths")

  @impl true
  def matvec(rows, vector) when is_list(rows) and is_list(vector) do
    Llamex.Tensor.matvec(rows, vector)
  end

  @impl true
  def matvec_pair(left_rows, right_rows, vector)
      when is_list(left_rows) and is_list(right_rows) and is_list(vector) do
    Llamex.Tensor.matvec_pair(left_rows, right_rows, vector)
  end

  @impl true
  def add(left, right) when is_list(left) and is_list(right) and length(left) == length(right) do
    left
    |> Enum.zip(right)
    |> Enum.map(fn {a, b} -> a + b end)
  end

  @impl true
  def argmax([first | rest]) do
    rest
    |> Enum.with_index(1)
    |> Enum.reduce({0, first}, fn {value, index}, {best_index, best_value} ->
      if value > best_value do
        {index, value}
      else
        {best_index, best_value}
      end
    end)
    |> elem(0)
  end

  @impl true
  def to_list(values) when is_list(values), do: values
end
