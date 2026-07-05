defmodule Llamex.Backend.FPGA do
  @moduledoc """
  FPGA-oriented backend boundary.

  The current implementation uses the List backend as a pure Elixir fallback.
  FPGA deployments can replace these functions with calls into the board
  runtime while keeping the engine and layer code unchanged.
  """

  @behaviour Llamex.Backend

  defdelegate from_list(values), to: Llamex.Backend.List
  defdelegate prepare_model(model), to: Llamex.Backend.List
  defdelegate dot(left, right), to: Llamex.Backend.List
  defdelegate matvec(rows, vector), to: Llamex.Backend.List
  defdelegate matvec_tensor(rows, vector), to: Llamex.Backend.List
  defdelegate matvec_pair(left_rows, right_rows, vector), to: Llamex.Backend.List
  defdelegate matvec_pair_tensor(left_rows, right_rows, vector), to: Llamex.Backend.List
  defdelegate matvec_split_pair_tensor(rows, left_count, vector), to: Llamex.Backend.List
  defdelegate matvec_triple(left_rows, middle_rows, right_rows, vector), to: Llamex.Backend.List

  defdelegate matvec_split_triple(rows, left_count, middle_count, right_count, vector),
    to: Llamex.Backend.List

  defdelegate silu_multiply(gate, up), to: Llamex.Backend.List
  defdelegate add(left, right), to: Llamex.Backend.List
  defdelegate argmax(tensor), to: Llamex.Backend.List
  defdelegate to_list(tensor), to: Llamex.Backend.List
end
