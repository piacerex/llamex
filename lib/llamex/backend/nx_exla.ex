defmodule Llamex.Backend.NxEXLA do
  @moduledoc """
  Optional Nx/EXLA-oriented tensor backend.

  This backend keeps Nx optional and can run with Nx's default backend. BEAM
  consumers that install EXLA can set Nx's default backend to EXLA before using
  this backend.
  """

  @behaviour Llamex.Backend

  @clients %{
    cpu: :host,
    host: :host,
    gpu: :cuda,
    cuda: :cuda,
    rocm: :rocm
  }
  @config_key {__MODULE__, :configured}

  @doc """
  Configures Nx to allocate tensors on EXLA for the selected target.

  Accepted targets are `:cpu`, `:host`, `:gpu`, `:cuda`, and `:rocm`.
  `:gpu` maps to CUDA; use `:rocm` explicitly for AMD ROCm.
  """
  def configure!(target) do
    client = client(target)
    nx = nx!()
    exla = exla!()

    validate_client_available!(client)

    apply(nx, :global_default_backend, [{Module.concat(exla, Backend), client: client}])

    if Code.ensure_loaded?(Nx.Defn) do
      apply(Nx.Defn, :global_default_options, [[compiler: exla, client: client]])
    end

    :persistent_term.put(@config_key, configured_info(target, client))

    :ok
  end

  def client(target) when is_binary(target) do
    target
    |> String.downcase()
    |> String.to_existing_atom()
    |> client()
  rescue
    ArgumentError -> raise ArgumentError, "unsupported EXLA target: #{inspect(target)}"
  end

  def client(target) when is_atom(target) do
    Map.fetch!(@clients, target)
  rescue
    KeyError -> raise ArgumentError, "unsupported EXLA target: #{inspect(target)}"
  end

  def info(target \\ :cpu) do
    target_client = client(target)

    %{
      nx_available?: Code.ensure_loaded?(Nx),
      exla_available?: Code.ensure_loaded?(EXLA) and Code.ensure_loaded?(EXLA.Backend),
      target: normalize_target(target),
      client: target_client,
      supported_platforms: supported_platforms(),
      target_available?: client_available?(target_client),
      xla_target: System.get_env("XLA_TARGET")
    }
  end

  def configured do
    :persistent_term.get(@config_key, nil)
  end

  @impl true
  def from_list(values) when is_list(values) do
    apply(nx!(), :tensor, [values, [type: {:f, 32}]])
  end

  @impl true
  def prepare_model(model) do
    model
    |> maybe_prepare_token_embeddings()
    |> Map.update!(:layers, &Enum.map(&1, fn layer -> prepare_layer(layer) end))
    |> Map.update(:output_norm, nil, &prepare_norm/1)
    |> Map.update!(:output, &prepare_output/1)
  end

  @impl true
  def dot(left, right) do
    result = apply(nx!(), :dot, [left, right])

    apply(nx!(), :to_number, [result])
  end

  @impl true
  def matvec(rows, vector) do
    nx = nx!()

    rows
    |> matvec_tensor(vector)
    |> then(&apply(nx, :to_flat_list, [&1]))
  end

  @impl true
  def matvec_tensor(rows, vector) do
    nx = nx!()
    matrix = tensor(rows)
    vector = tensor(vector)

    apply(nx, :dot, [matrix, vector])
  end

  @impl true
  def top_k_matvec(rows, vector, top_k, opts)
      when is_integer(top_k) and top_k > 0 and is_list(opts) do
    nx = nx!()
    vocab_size = row_count(rows)

    logits =
      rows
      |> matvec_tensor(vector)
      |> apply_repetition_penalty_tensor(opts, vocab_size, nx)
      |> suppress_tokens_tensor(opts, vocab_size, nx)

    {values, indices} = apply(nx, :top_k, [logits, [k: min(top_k, vocab_size)]])

    values = apply(nx, :to_flat_list, [values])
    indices = apply(nx, :to_flat_list, [indices])

    Enum.zip(values, indices)
  end

  @impl true
  def rope(vector, position, theta, dimension_count)
      when is_integer(position) and position >= 0 and is_number(theta) do
    nx = nx!()
    vector = tensor(vector)
    vector_size = vector |> shape() |> elem(0)
    dimension_count = dimension_count || vector_size - rem(vector_size, 2)

    cond do
      dimension_count == 0 ->
        to_list(vector)

      dimension_count > vector_size ->
        raise ArgumentError, "RoPE dimension count cannot exceed vector length"

      rem(dimension_count, 2) != 0 ->
        raise ArgumentError, "RoPE vector length must be even"

      true ->
        vector
        |> apply_rope_tensor(position, theta, dimension_count, nx)
        |> to_list()
    end
  end

  @impl true
  def matvec_pair(left_rows, right_rows, vector) do
    nx = nx!()

    {gate, up} = matvec_pair_tensor(left_rows, right_rows, vector)

    {apply(nx, :to_flat_list, [gate]), apply(nx, :to_flat_list, [up])}
  end

  @impl true
  def matvec_pair_tensor(left_rows, right_rows, vector) do
    left_count = row_count(left_rows)
    matrix = concatenate_rows([left_rows, right_rows])

    matvec_split_pair_tensor(matrix, left_count, vector)
  end

  @impl true
  def matvec_split_pair_tensor(rows, left_count, vector) do
    nx = nx!()
    values = matvec_tensor(rows, vector)
    right_count = row_count(rows) - left_count

    {apply(nx, :slice, [values, [0], [left_count]]),
     apply(nx, :slice, [values, [left_count], [right_count]])}
  end

  @impl true
  def silu_multiply(gate, up) do
    nx = nx!()

    apply(nx, :multiply, [apply_silu(gate, nx), up])
  end

  @impl true
  def rms_norm(input, weight, epsilon) do
    nx = nx!()
    input = tensor(input)
    weight = tensor(weight)

    mean_square =
      input
      |> then(&apply(nx, :multiply, [&1, &1]))
      |> then(&apply(nx, :mean, [&1]))

    scale =
      mean_square
      |> then(&apply(nx, :add, [&1, epsilon]))
      |> then(&apply(nx, :sqrt, [&1]))
      |> then(&apply(nx, :divide, [1.0, &1]))

    input
    |> then(&apply(nx, :multiply, [&1, scale]))
    |> then(&apply(nx, :multiply, [&1, weight]))
  end

  @impl true
  def attend_head(query, keys, values) do
    query
    |> attend_head_tensors(tensor(keys), tensor(values))
    |> to_list()
  end

  @impl true
  def attend_heads(query_heads, entries, head_count, kv_head_count)
      when is_list(query_heads) and is_list(entries) and is_integer(head_count) and
             head_count > 0 and is_integer(kv_head_count) and kv_head_count > 0 do
    cache = grouped_kv_tensors(entries, kv_head_count)

    heads =
      query_heads
      |> Enum.with_index()
      |> Enum.map(fn {query, head_index} ->
        {keys, values} = Map.fetch!(cache, div(head_index * kv_head_count, head_count))
        attend_head_tensors(query, keys, values)
      end)

    apply(nx!(), :concatenate, [heads, [axis: 0]])
  end

  @impl true
  def matvec_triple(left_rows, middle_rows, right_rows, vector) do
    left_count = row_count(left_rows)
    middle_count = row_count(middle_rows)
    right_count = row_count(right_rows)
    matrix = concatenate_rows([left_rows, middle_rows, right_rows])

    matvec_split_triple(matrix, left_count, middle_count, right_count, vector)
  end

  @impl true
  def matvec_split_triple(rows, left_count, middle_count, right_count, vector) do
    nx = nx!()

    values =
      rows
      |> matvec_tensor(vector)
      |> then(&apply(nx, :to_flat_list, [&1]))

    {left, rest} = Enum.split(values, left_count)
    {middle, right} = Enum.split(rest, middle_count)

    if length(right) != right_count do
      raise ArgumentError, "matvec_triple split produced an unexpected row count"
    end

    {left, middle, right}
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
      when is_integer(head_count) and head_count > 0 and is_integer(kv_head_count) and
             kv_head_count > 0 do
    nx = nx!()

    values = matvec_tensor(weight, input)
    query = apply(nx, :slice, [values, [0], [q_count]])
    key = apply(nx, :slice, [values, [q_count], [k_count]])
    value = apply(nx, :slice, [values, [q_count + k_count], [v_count]])

    query_heads =
      query
      |> reshape_tensor_heads(head_count, nx)
      |> rope_head_matrix(position, rope_theta, rope_dimension_count, nx)
      |> split_head_matrix(head_count, nx)

    key_heads =
      key
      |> reshape_tensor_heads(kv_head_count, nx)
      |> rope_head_matrix(position, rope_theta, rope_dimension_count, nx)
      |> split_head_matrix(kv_head_count, nx)

    value_heads =
      value
      |> reshape_tensor_heads(kv_head_count, nx)
      |> split_head_matrix(kv_head_count, nx)

    {query_heads, key_heads, value_heads}
  end

  @impl true
  def add(left, right), do: apply(nx!(), :add, [tensor(left), tensor(right)])

  @impl true
  def argmax(tensor) do
    result = apply(nx!(), :argmax, [tensor])

    apply(nx!(), :to_number, [result])
  end

  @impl true
  def to_list(values) when is_list(values), do: values
  def to_list(tensor), do: apply(nx!(), :to_flat_list, [tensor])

  defp nx! do
    if Code.ensure_loaded?(Nx) do
      Nx
    else
      raise "Nx is not available; add {:nx, ...} and select an EXLA compiler if needed"
    end
  end

  defp exla! do
    if Code.ensure_loaded?(EXLA) and Code.ensure_loaded?(EXLA.Backend) do
      EXLA
    else
      raise "EXLA is not available; add {:exla, ...} and run mix deps.get"
    end
  end

  defp supported_platforms do
    if Code.ensure_loaded?(EXLA.Client) and
         function_exported?(EXLA.Client, :get_supported_platforms, 0) do
      EXLA.Client.get_supported_platforms()
    else
      %{}
    end
  rescue
    _exception -> %{}
  end

  defp configured_info(target, client) do
    %{
      target: normalize_target(target),
      client: client,
      target_available?: client_available?(client),
      xla_target: System.get_env("XLA_TARGET")
    }
  end

  defp validate_client_available!(client) do
    if client_available?(client) do
      :ok
    else
      raise "EXLA client #{inspect(client)} is not available; supported platforms: #{format_platforms(supported_platforms())}"
    end
  end

  defp client_available?(client) do
    platforms = supported_platforms()
    platforms == %{} or Map.has_key?(platforms, client)
  end

  defp format_platforms(platforms) when map_size(platforms) == 0, do: "unknown"

  defp format_platforms(platforms) do
    platforms
    |> Map.keys()
    |> Enum.sort()
    |> Enum.map_join(", ", &to_string/1)
  end

  defp normalize_target(target) when is_binary(target) do
    target
    |> String.downcase()
    |> String.to_existing_atom()
  end

  defp normalize_target(target) when is_atom(target), do: target

  defp maybe_prepare_token_embeddings(%{token_embeddings: token_embeddings} = model) do
    %{model | token_embeddings: prepare_token_embeddings(token_embeddings)}
  end

  defp maybe_prepare_token_embeddings(model), do: model

  defp prepare_token_embeddings(token_embeddings) do
    Map.new(token_embeddings, fn {token, embedding} -> {token, tensor(embedding)} end)
  end

  defp prepare_layer(layer) do
    [:attention_norm, :feed_forward_norm, :wq, :wk, :wv, :wo, :w_gate, :w_up, :w_down]
    |> Enum.reduce(layer, fn key, layer ->
      case Map.fetch(layer, key) do
        {:ok, weights} -> Map.put(layer, key, tensor(weights))
        :error -> layer
      end
    end)
    |> maybe_prepare_combined(:w_qkv, [:wq, :wk, :wv])
    |> maybe_prepare_combined(:w_gate_up, [:w_gate, :w_up])
  end

  defp prepare_output(nil), do: nil

  defp prepare_output(%{weight: _weight} = output) do
    output
    |> Map.update!(:weight, &tensor/1)
    |> Map.update(:norm, nil, &tensor/1)
  end

  defp prepare_norm(nil), do: nil
  defp prepare_norm(weight), do: tensor(weight)

  defp row_count(rows) when is_list(rows), do: length(rows)
  defp row_count(rows), do: rows |> shape() |> elem(0)

  defp shape(tensor), do: apply(nx!(), :shape, [tensor])

  defp apply_silu(tensor, nx) do
    denominator =
      tensor
      |> then(&apply(nx, :negate, [&1]))
      |> then(&apply(nx, :exp, [&1]))
      |> then(&apply(nx, :add, [&1, 1.0]))

    apply(nx, :divide, [tensor, denominator])
  end

  defp apply_rope_tensor(vector, position, theta, dimension_count, nx) do
    half = div(dimension_count, 2)
    left = apply(nx, :slice, [vector, [0], [half]])
    right = apply(nx, :slice, [vector, [half], [half]])
    pass_count = (vector |> shape() |> elem(0)) - dimension_count
    angles = rope_angles(position, theta, dimension_count, half)
    cos = angles |> Enum.map(&:math.cos/1) |> tensor()
    sin = angles |> Enum.map(&:math.sin/1) |> tensor()

    rotated_left =
      apply(nx, :subtract, [
        apply(nx, :multiply, [left, cos]),
        apply(nx, :multiply, [right, sin])
      ])

    rotated_right =
      apply(nx, :add, [
        apply(nx, :multiply, [left, sin]),
        apply(nx, :multiply, [right, cos])
      ])

    parts =
      if pass_count > 0 do
        [
          rotated_left,
          rotated_right,
          apply(nx, :slice, [vector, [dimension_count], [pass_count]])
        ]
      else
        [rotated_left, rotated_right]
      end

    apply(nx, :concatenate, [parts, [axis: 0]])
  end

  defp grouped_kv_tensors(entries, kv_head_count) do
    0..(kv_head_count - 1)
    |> Map.new(fn head_index ->
      keys =
        Enum.map(entries, fn {cached_keys, _cached_values} ->
          Enum.at(cached_keys, head_index)
        end)

      values =
        Enum.map(entries, fn {_cached_keys, cached_values} ->
          Enum.at(cached_values, head_index)
        end)

      {head_index, {stack_tensors(keys), stack_tensors(values)}}
    end)
  end

  defp stack_tensors(tensors) do
    apply(nx!(), :stack, [Enum.map(tensors, &tensor/1), [axis: 0]])
  end

  defp reshape_tensor_heads(vector, head_count, nx) do
    size = vector |> shape() |> elem(0)

    if rem(size, head_count) != 0 do
      raise ArgumentError, "vector length must be divisible by split size"
    end

    apply(nx, :reshape, [vector, {head_count, div(size, head_count)}])
  end

  defp rope_head_matrix(heads, position, theta, dimension_count, nx) do
    {_head_count, head_size} = shape(heads)
    dimension_count = dimension_count || head_size - rem(head_size, 2)

    cond do
      dimension_count == 0 ->
        heads

      dimension_count > head_size ->
        raise ArgumentError, "RoPE dimension count cannot exceed vector length"

      rem(dimension_count, 2) != 0 ->
        raise ArgumentError, "RoPE vector length must be even"

      true ->
        apply_rope_head_matrix(heads, position, theta, dimension_count, nx)
    end
  end

  defp apply_rope_head_matrix(heads, position, theta, dimension_count, nx) do
    {head_count, head_size} = shape(heads)
    half = div(dimension_count, 2)
    left = apply(nx, :slice, [heads, [0, 0], [head_count, half]])
    right = apply(nx, :slice, [heads, [0, half], [head_count, half]])
    pass_count = head_size - dimension_count
    angles = rope_angles(position, theta, dimension_count, half)
    cos = angles |> Enum.map(&:math.cos/1) |> tensor()
    sin = angles |> Enum.map(&:math.sin/1) |> tensor()

    rotated_left =
      apply(nx, :subtract, [
        apply(nx, :multiply, [left, cos]),
        apply(nx, :multiply, [right, sin])
      ])

    rotated_right =
      apply(nx, :add, [
        apply(nx, :multiply, [left, sin]),
        apply(nx, :multiply, [right, cos])
      ])

    parts =
      if pass_count > 0 do
        [
          rotated_left,
          rotated_right,
          apply(nx, :slice, [heads, [0, dimension_count], [head_count, pass_count]])
        ]
      else
        [rotated_left, rotated_right]
      end

    apply(nx, :concatenate, [parts, [axis: 1]])
  end

  defp split_head_matrix(heads, head_count, nx) do
    {_head_count, head_size} = shape(heads)

    Enum.map(0..(head_count - 1), fn head_index ->
      heads
      |> then(&apply(nx, :slice, [&1, [head_index, 0], [1, head_size]]))
      |> then(&apply(nx, :reshape, [&1, {head_size}]))
    end)
  end

  defp attend_head_tensors(query, keys, values) do
    nx = nx!()
    query = tensor(query)
    scale = 1.0 / :math.sqrt(query |> shape() |> elem(0))

    weights =
      keys
      |> then(&apply(nx, :dot, [&1, query]))
      |> then(&apply(nx, :multiply, [&1, scale]))
      |> softmax(nx)

    weights
    |> then(&apply(nx, :dot, [&1, values]))
  end

  defp rope_angles(position, theta, dimension_count, half) do
    Enum.map(0..(half - 1), fn pair_index ->
      position / :math.pow(theta, 2 * pair_index / dimension_count)
    end)
  end

  defp softmax(values, nx) do
    exps =
      values
      |> then(&apply(nx, :subtract, [&1, apply(nx, :reduce_max, [&1])]))
      |> then(&apply(nx, :exp, [&1]))

    apply(nx, :divide, [exps, apply(nx, :sum, [exps])])
  end

  defp apply_repetition_penalty_tensor(logits, opts, vocab_size, nx) do
    penalty = Keyword.get(opts, :repetition_penalty)
    history = Keyword.get(opts, :history, [])

    if is_number(penalty) and penalty > 0.0 and history != [] do
      repeated =
        history
        |> index_mask_tensor(vocab_size, 1.0, nx)
        |> then(&apply(nx, :equal, [&1, 1.0]))

      positive = apply(nx, :greater_equal, [logits, 0.0])

      penalized =
        apply(nx, :select, [
          positive,
          apply(nx, :divide, [logits, penalty]),
          apply(nx, :multiply, [logits, penalty])
        ])

      apply(nx, :select, [repeated, penalized, logits])
    else
      logits
    end
  end

  defp suppress_tokens_tensor(logits, opts, vocab_size, nx) do
    suppressed = Keyword.get(opts, :suppress_tokens, [])

    if is_list(suppressed) and suppressed != [] do
      mask = index_mask_tensor(suppressed, vocab_size, -1.0e30, nx)
      apply(nx, :add, [logits, tensor(mask)])
    else
      logits
    end
  end

  defp index_mask_tensor(indices, size, value, nx) do
    indices = Enum.filter(indices, &(&1 >= 0 and &1 < size))

    if indices == [] do
      tensor(List.duplicate(0.0, size))
    else
      target = 0.0 |> tensor() |> then(&apply(nx, :broadcast, [&1, {size}]))
      updates = tensor(List.duplicate(value, length(indices)))
      indexed = indices |> Enum.map(&[&1]) |> int_tensor()

      apply(nx, :indexed_put, [target, indexed, updates])
    end
  end

  defp maybe_prepare_combined(layer, combined_key, keys) do
    if Enum.all?(keys, &Map.has_key?(layer, &1)) do
      weights = Enum.map(keys, &Map.fetch!(layer, &1))

      layer
      |> Map.put(combined_key, concatenate_rows(weights))
      |> Map.put(:"#{combined_key}_row_counts", Enum.map(weights, &row_count/1))
    else
      layer
    end
  end

  defp concatenate_rows(rows) do
    apply(nx!(), :concatenate, [Enum.map(rows, &tensor/1), [axis: 0]])
  end

  defp tensor(values) when is_list(values), do: apply(nx!(), :tensor, [values, [type: {:f, 32}]])
  defp tensor(value), do: value

  defp int_tensor(values) when is_list(values),
    do: apply(nx!(), :tensor, [values, [type: {:s, 32}]])
end
