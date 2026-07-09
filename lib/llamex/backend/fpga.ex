defmodule Llamex.Backend.FPGA do
  @moduledoc """
  FPGA-oriented backend boundary.

  The current implementation uses the List backend as a pure Elixir fallback.
  FPGA deployments can replace these functions with calls into the board
  runtime while keeping the engine and layer code unchanged.
  """

  @behaviour Llamex.Backend

  @app :llamex
  @runtime_key :fpga_runtime

  def target, do: :fpga

  def fallback_backend, do: Llamex.Backend.List

  def configure_runtime!(runtime) when is_atom(runtime) do
    Application.put_env(@app, @runtime_key, runtime)
  end

  def clear_runtime! do
    Application.delete_env(@app, @runtime_key)
  end

  def runtime, do: Application.get_env(@app, @runtime_key)

  def capabilities do
    runtime = runtime()

    %{
      target: target(),
      status: if(runtime, do: :delegated, else: :fallback),
      runtime_backend: runtime,
      fallback_backend: fallback_backend(),
      tensor_format: :dequantized,
      atomvm_oriented?: true
    }
  end

  def from_list(values), do: dispatch(:from_list, [values])
  def prepare_model(model), do: dispatch(:prepare_model, [model])
  def dot(left, right), do: dispatch(:dot, [left, right])
  def matvec(rows, vector), do: dispatch(:matvec, [rows, vector])
  def matvec_tensor(rows, vector), do: dispatch(:matvec_tensor, [rows, vector])

  def top_k_matvec(rows, vector, top_k, opts),
    do: dispatch(:top_k_matvec, [rows, vector, top_k, opts])

  def rope(vector, position, theta, dimension_count),
    do: dispatch(:rope, [vector, position, theta, dimension_count])

  def matvec_pair(left_rows, right_rows, vector),
    do: dispatch(:matvec_pair, [left_rows, right_rows, vector])

  def matvec_pair_tensor(left_rows, right_rows, vector),
    do: dispatch(:matvec_pair_tensor, [left_rows, right_rows, vector])

  def matvec_split_pair_tensor(rows, left_count, vector),
    do: dispatch(:matvec_split_pair_tensor, [rows, left_count, vector])

  def matvec_triple(left_rows, middle_rows, right_rows, vector),
    do: dispatch(:matvec_triple, [left_rows, middle_rows, right_rows, vector])

  def matvec_split_triple(rows, left_count, middle_count, right_count, vector),
    do: dispatch(:matvec_split_triple, [rows, left_count, middle_count, right_count, vector])

  def qkv_heads(
        weight,
        counts,
        input,
        head_count,
        kv_head_count,
        position,
        rope_theta,
        rope_dimension_count
      ) do
    dispatch(:qkv_heads, [
      weight,
      counts,
      input,
      head_count,
      kv_head_count,
      position,
      rope_theta,
      rope_dimension_count
    ])
  end

  def silu_multiply(gate, up), do: dispatch(:silu_multiply, [gate, up])
  def rms_norm(input, weight, epsilon), do: dispatch(:rms_norm, [input, weight, epsilon])
  def attend_head(query, keys, values), do: dispatch(:attend_head, [query, keys, values])
  def prepare_kv_entries(entries), do: dispatch(:prepare_kv_entries, [entries])
  def append_kv_entry(entries, key, value), do: dispatch(:append_kv_entry, [entries, key, value])

  def attend_heads(query_heads, entries, head_count, kv_head_count),
    do: dispatch(:attend_heads, [query_heads, entries, head_count, kv_head_count])

  def add(left, right), do: dispatch(:add, [left, right])
  def argmax(tensor), do: dispatch(:argmax, [tensor])
  def to_list(tensor), do: dispatch(:to_list, [tensor])

  defp dispatch(function, args) do
    runtime = runtime()

    if runtime && function_exported?(runtime, function, length(args)) do
      apply(runtime, function, args)
    else
      apply(fallback_backend(), function, args)
    end
  end
end
