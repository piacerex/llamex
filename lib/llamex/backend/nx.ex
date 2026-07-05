defmodule Llamex.Backend.Nx do
  @moduledoc """
  Backward-compatible alias for `Llamex.Backend.NxEXLA`.
  """

  defdelegate from_list(values), to: Llamex.Backend.NxEXLA
  defdelegate prepare_model(model), to: Llamex.Backend.NxEXLA
  defdelegate dot(left, right), to: Llamex.Backend.NxEXLA
  defdelegate matvec(rows, vector), to: Llamex.Backend.NxEXLA
  defdelegate matvec_pair(left_rows, right_rows, vector), to: Llamex.Backend.NxEXLA
  defdelegate add(left, right), to: Llamex.Backend.NxEXLA
  defdelegate argmax(tensor), to: Llamex.Backend.NxEXLA
  defdelegate to_list(tensor), to: Llamex.Backend.NxEXLA
end
