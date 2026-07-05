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
  def matvec_pair(left_rows, right_rows, vector) do
    nx = nx!()

    {gate, up} = matvec_pair_tensor(left_rows, right_rows, vector)

    {apply(nx, :to_flat_list, [gate]), apply(nx, :to_flat_list, [up])}
  end

  @impl true
  def matvec_pair_tensor(left_rows, right_rows, vector) do
    nx = nx!()
    left_count = row_count(left_rows)
    matrix = apply(nx, :concatenate, [[tensor(left_rows), tensor(right_rows)], [axis: 0]])
    vector = tensor(vector)

    values = apply(nx, :dot, [matrix, vector])

    {
      apply(nx, :slice, [values, [0], [left_count]]),
      apply(nx, :slice, [values, [left_count], [left_count]])
    }
  end

  @impl true
  def silu_multiply(gate, up) do
    nx = nx!()

    apply(nx, :multiply, [apply_silu(gate, nx), up])
  end

  @impl true
  def matvec_triple(left_rows, middle_rows, right_rows, vector) do
    nx = nx!()
    left_count = row_count(left_rows)
    middle_count = row_count(middle_rows)
    right_count = row_count(right_rows)

    matrix =
      apply(nx, :concatenate, [
        [tensor(left_rows), tensor(middle_rows), tensor(right_rows)],
        [axis: 0]
      ])

    values =
      matrix
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

  defp apply_silu(tensor, nx) do
    denominator =
      tensor
      |> then(&apply(nx, :negate, [&1]))
      |> then(&apply(nx, :exp, [&1]))
      |> then(&apply(nx, :add, [&1, 1.0]))

    apply(nx, :divide, [tensor, denominator])
  end

  defp tensor(values) when is_list(values), do: apply(nx!(), :tensor, [values, [type: {:f, 32}]])
  defp tensor(value), do: value
end
