defmodule Llamex.TensorStore do
  @moduledoc """
  Named tensor decoding for model files.
  """

  @supported_dtypes MapSet.new(["f32", "f16"])

  def decode(tensors) when is_map(tensors) do
    Map.new(tensors, fn {name, tensor} ->
      {name, decode_tensor(name, tensor)}
    end)
  end

  def fetch_matrix(tensors, name) when is_map(tensors) and is_binary(name) do
    tensors
    |> Map.fetch!(name)
    |> Map.fetch!(:matrix)
  end

  defp decode_tensor(name, %{"shape" => shape, "dtype" => dtype, "data" => data})
       when is_binary(name) and is_list(shape) and is_binary(dtype) and is_list(data) do
    validate_dtype!(name, dtype)
    validate_shape!(name, shape)
    validate_data_size!(name, shape, data)

    %{shape: shape, dtype: dtype, data: data, matrix: to_matrix(shape, data)}
  end

  defp decode_tensor(name, _tensor) do
    raise ArgumentError, "tensor #{name} must contain shape, dtype, and data"
  end

  defp validate_dtype!(name, dtype) do
    if not MapSet.member?(@supported_dtypes, dtype) do
      raise ArgumentError, "tensor #{name} has unsupported dtype #{dtype}"
    end
  end

  defp validate_shape!(name, shape) do
    if shape == [] or Enum.any?(shape, &(&1 <= 0)) do
      raise ArgumentError, "tensor #{name} shape must contain positive dimensions"
    end
  end

  defp validate_data_size!(name, shape, data) do
    expected = Enum.product(shape)

    if length(data) != expected do
      raise ArgumentError, "tensor #{name} data length must match shape product"
    end
  end

  defp to_matrix([rows, columns], data) do
    data
    |> Enum.chunk_every(columns)
    |> then(fn matrix ->
      if length(matrix) != rows do
        raise ArgumentError, "matrix row count does not match shape"
      end

      matrix
    end)
  end

  defp to_matrix(_shape, _data) do
    raise ArgumentError, "only rank-2 tensors can be converted to matrix"
  end
end
