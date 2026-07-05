defmodule Llamex.Backend.Nx do
  @moduledoc """
  Backward-compatible alias for `Llamex.Backend.NxEXLA`.
  """

  defdelegate configure!(target), to: Llamex.Backend.NxEXLA
  defdelegate client(target), to: Llamex.Backend.NxEXLA
  defdelegate from_list(values), to: Llamex.Backend.NxEXLA
  defdelegate prepare_model(model), to: Llamex.Backend.NxEXLA
  defdelegate dot(left, right), to: Llamex.Backend.NxEXLA
  defdelegate matvec(rows, vector), to: Llamex.Backend.NxEXLA
  defdelegate matvec_tensor(rows, vector), to: Llamex.Backend.NxEXLA
  defdelegate top_k_matvec(rows, vector, top_k, opts), to: Llamex.Backend.NxEXLA
  defdelegate rope(vector, position, theta, dimension_count), to: Llamex.Backend.NxEXLA
  defdelegate matvec_pair(left_rows, right_rows, vector), to: Llamex.Backend.NxEXLA
  defdelegate matvec_pair_tensor(left_rows, right_rows, vector), to: Llamex.Backend.NxEXLA
  defdelegate matvec_split_pair_tensor(rows, left_count, vector), to: Llamex.Backend.NxEXLA
  defdelegate matvec_triple(left_rows, middle_rows, right_rows, vector), to: Llamex.Backend.NxEXLA

  defdelegate matvec_split_triple(rows, left_count, middle_count, right_count, vector),
    to: Llamex.Backend.NxEXLA

  defdelegate silu_multiply(gate, up), to: Llamex.Backend.NxEXLA
  defdelegate rms_norm(input, weight, epsilon), to: Llamex.Backend.NxEXLA
  defdelegate attend_head(query, keys, values), to: Llamex.Backend.NxEXLA

  defdelegate attend_heads(query_heads, entries, head_count, kv_head_count),
    to: Llamex.Backend.NxEXLA

  defdelegate add(left, right), to: Llamex.Backend.NxEXLA
  defdelegate argmax(tensor), to: Llamex.Backend.NxEXLA
  defdelegate to_list(tensor), to: Llamex.Backend.NxEXLA
end
