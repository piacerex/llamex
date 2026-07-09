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
  def matvec_tensor(rows, vector) when is_list(rows) and is_list(vector) do
    matvec(rows, vector)
  end

  def matvec_tensor(%{info: %{type_name: "Q4_0"}, payload: payload} = compact, vector)
      when is_binary(payload) and is_list(vector) do
    compact
    |> Llamex.TensorStore.dequantize_compact_matrix()
    |> matvec(vector)
  end

  @impl true
  def top_k_matvec(rows, vector, top_k, opts)
      when is_list(rows) and is_list(vector) and is_integer(top_k) and top_k > 0 and
             is_list(opts) do
    Llamex.Tensor.top_k_matvec(rows, vector, top_k, opts)
  end

  @impl true
  def rope(vector, position, theta, dimension_count)
      when is_list(vector) and is_integer(position) and position >= 0 and is_number(theta) do
    Llamex.Layers.RoPE.apply(vector, position, theta, dimension_count)
  end

  @impl true
  def matvec_pair(left_rows, right_rows, vector)
      when is_list(left_rows) and is_list(right_rows) and is_list(vector) do
    Llamex.Tensor.matvec_pair(left_rows, right_rows, vector)
  end

  @impl true
  def matvec_pair_tensor(left_rows, right_rows, vector)
      when is_list(left_rows) and is_list(right_rows) and is_list(vector) do
    matvec_pair(left_rows, right_rows, vector)
  end

  def matvec_pair_tensor(left_rows, right_rows, vector) when is_list(vector) do
    {matvec_tensor(left_rows, vector), matvec_tensor(right_rows, vector)}
  end

  @impl true
  def matvec_split_pair_tensor(rows, left_count, vector)
      when is_list(rows) and is_integer(left_count) and left_count > 0 and is_list(vector) do
    {left_rows, right_rows} = Enum.split(rows, left_count)

    matvec_pair_tensor(left_rows, right_rows, vector)
  end

  @impl true
  def silu_multiply(gate, up) when is_list(gate) and is_list(up) do
    gate
    |> Llamex.Tensor.silu()
    |> Llamex.Tensor.multiply(up)
  end

  @impl true
  def rms_norm(input, weight, epsilon)
      when is_list(input) and is_list(weight) and length(input) == length(weight) do
    Llamex.Layers.RMSNorm.forward(input, weight, epsilon)
  end

  @impl true
  def attend_head(query, keys, values)
      when is_list(query) and is_list(keys) and is_list(values) do
    scale = 1.0 / :math.sqrt(length(query))

    keys
    |> Enum.map(&(Llamex.Tensor.dot(query, &1) * scale))
    |> Llamex.Tensor.softmax()
    |> Llamex.Tensor.weighted_sum(values)
  end

  @impl true
  def prepare_kv_entries(entries) when is_list(entries), do: entries

  @impl true
  def append_kv_entry(entries, key, value)
      when is_list(entries) and is_list(key) and is_list(value) do
    entries ++ [{key, value}]
  end

  @impl true
  def attend_heads(query_heads, entries, head_count, 1)
      when is_list(query_heads) and is_list(entries) and is_integer(head_count) and head_count > 0 do
    keys =
      Enum.map(entries, fn {cached_keys, _cached_values} ->
        hd(cached_keys)
      end)

    values =
      Enum.map(entries, fn {_cached_keys, cached_values} ->
        hd(cached_values)
      end)

    Enum.flat_map(query_heads, &attend_head(&1, keys, values))
  end

  @impl true
  def attend_heads(query_heads, entries, head_count, kv_head_count)
      when is_list(query_heads) and is_list(entries) and is_integer(head_count) and
             head_count > 0 and is_integer(kv_head_count) and kv_head_count > 0 do
    query_heads
    |> Enum.with_index()
    |> Enum.flat_map(fn {query, head_index} ->
      kv_head_index = div(head_index * kv_head_count, head_count)

      keys =
        Enum.map(entries, fn {cached_keys, _cached_values} ->
          Enum.at(cached_keys, kv_head_index)
        end)

      values =
        Enum.map(entries, fn {_cached_keys, cached_values} ->
          Enum.at(cached_values, kv_head_index)
        end)

      attend_head(query, keys, values)
    end)
  end

  @impl true
  def matvec_triple(left_rows, middle_rows, right_rows, vector)
      when is_list(left_rows) and is_list(middle_rows) and is_list(right_rows) and is_list(vector) do
    {matvec(left_rows, vector), matvec(middle_rows, vector), matvec(right_rows, vector)}
  end

  def matvec_triple(left_rows, middle_rows, right_rows, vector) when is_list(vector) do
    {
      matvec_tensor(left_rows, vector),
      matvec_tensor(middle_rows, vector),
      matvec_tensor(right_rows, vector)
    }
  end

  @impl true
  def matvec_split_triple(rows, left_count, middle_count, right_count, vector)
      when is_list(rows) and is_integer(left_count) and left_count > 0 and
             is_integer(middle_count) and middle_count > 0 and is_integer(right_count) and
             right_count > 0 and is_list(vector) do
    {left_rows, rest} = Enum.split(rows, left_count)
    {middle_rows, right_rows} = Enum.split(rest, middle_count)

    if length(right_rows) != right_count do
      raise ArgumentError, "matvec_split_triple split produced an unexpected row count"
    end

    matvec_triple(left_rows, middle_rows, right_rows, vector)
  end

  @impl true
  def qkv_heads(
        weight,
        [q_count, k_count, v_count],
        input,
        head_count,
        kv_head_count,
        position,
        rope_theta,
        rope_dimension_count
      )
      when is_list(weight) and is_list(input) and is_integer(head_count) and head_count > 0 and
             is_integer(kv_head_count) and kv_head_count > 0 do
    {query, key, value} = matvec_split_triple(weight, q_count, k_count, v_count, input)

    query_heads =
      query
      |> split_heads(head_count)
      |> Enum.map(&rope(&1, position, rope_theta, rope_dimension_count))

    key_heads =
      key
      |> split_heads(kv_head_count)
      |> Enum.map(&rope(&1, position, rope_theta, rope_dimension_count))

    {query_heads, key_heads, split_heads(value, kv_head_count)}
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

  defp split_heads(vector, head_count) do
    Llamex.Tensor.split_every(vector, div(length(vector), head_count))
  end
end
