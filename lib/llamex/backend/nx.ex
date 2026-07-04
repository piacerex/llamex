defmodule Llamex.Backend.Nx do
  @moduledoc """
  Optional Nx-backed tensor operations.

  The module does not make Nx a hard dependency. Callers should only select this
  backend in environments where Nx is available.
  """

  @behaviour Llamex.Backend

  @impl true
  def from_list(values) when is_list(values) do
    apply(nx!(), :tensor, [values, [type: {:f, 32}]])
  end

  @impl true
  def dot(left, right) do
    result = apply(nx!(), :dot, [left, right])

    apply(nx!(), :to_number, [result])
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
end
