defmodule Llamex.Backend.NxEXLA do
  @moduledoc """
  Optional Nx/EXLA-oriented tensor backend.

  This backend keeps Nx optional and can run with Nx's default backend. BEAM
  consumers that install EXLA can set Nx's default backend to EXLA before using
  this backend.
  """

  @behaviour Llamex.Backend

  @impl true
  def from_list(values) when is_list(values) do
    apply(nx!(), :tensor, [values, [type: {:f, 32}]])
  end

  @impl true
  def prepare_model(model) do
    %{
      model
      | layers: Enum.map(model.layers, &prepare_layer/1),
        output: prepare_output(model.output)
    }
  end

  @impl true
  def dot(left, right) do
    result = apply(nx!(), :dot, [left, right])

    apply(nx!(), :to_number, [result])
  end

  @impl true
  def matvec(rows, vector) when is_list(vector) do
    nx = nx!()
    matrix = tensor(rows)
    vector = apply(nx, :tensor, [vector, [type: {:f, 32}]])

    nx
    |> apply(:dot, [matrix, vector])
    |> then(&apply(nx, :to_flat_list, [&1]))
  end

  @impl true
  def matvec_pair(left_rows, right_rows, vector) when is_list(vector) do
    nx = nx!()
    left_count = row_count(left_rows)
    matrix = apply(nx, :concatenate, [[tensor(left_rows), tensor(right_rows)], [axis: 0]])
    vector = apply(nx, :tensor, [vector, [type: {:f, 32}]])

    values =
      nx
      |> apply(:dot, [matrix, vector])
      |> then(&apply(nx, :to_flat_list, [&1]))

    Enum.split(values, left_count)
  end

  @impl true
  def add(left, right), do: apply(nx!(), :add, [left, right])

  @impl true
  def argmax(tensor) do
    result = apply(nx!(), :argmax, [tensor])

    apply(nx!(), :to_number, [result])
  end

  @impl true
  def to_list(tensor), do: apply(nx!(), :to_flat_list, [tensor])

  defp nx! do
    if Code.ensure_loaded?(Nx) do
      Nx
    else
      raise "Nx is not available; add {:nx, ...} and select an EXLA compiler if needed"
    end
  end

  defp prepare_layer(layer) do
    [:wq, :wk, :wv, :wo, :w_gate, :w_up, :w_down]
    |> Enum.reduce(layer, fn key, layer ->
      case Map.fetch(layer, key) do
        {:ok, weights} -> Map.put(layer, key, tensor(weights))
        :error -> layer
      end
    end)
  end

  defp prepare_output(nil), do: nil

  defp prepare_output(%{weight: weight} = output) do
    %{output | weight: tensor(weight)}
  end

  defp row_count(rows) when is_list(rows), do: length(rows)
  defp row_count(rows), do: rows |> shape() |> elem(0)

  defp shape(tensor), do: apply(nx!(), :shape, [tensor])

  defp tensor(values) when is_list(values), do: apply(nx!(), :tensor, [values, [type: {:f, 32}]])
  defp tensor(value), do: value
end
