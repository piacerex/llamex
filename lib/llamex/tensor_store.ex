defmodule Llamex.TensorStore do
  @moduledoc """
  Named tensor decoding for model files.
  """

  @supported_dtypes MapSet.new(["f32", "f16"])
  @q4_0_block_size 32

  def decode(tensors) when is_map(tensors) do
    Map.new(tensors, fn {name, tensor} ->
      {name, decode_tensor(name, tensor)}
    end)
  end

  def compact_tensor?(%{"payload" => payload, "quantized?" => quantized?})
      when is_binary(payload) and is_boolean(quantized?),
      do: true

  def compact_tensor?(_tensor), do: false

  def compact_tensor_info(%{"shape" => shape, "dtype" => dtype} = tensor)
      when is_list(shape) and is_binary(dtype) do
    unless compact_tensor?(tensor) do
      raise ArgumentError, "tensor is not a compact GGUF payload"
    end

    %{
      shape: shape,
      dtype: dtype,
      type: Map.get(tensor, "type"),
      type_name: Map.get(tensor, "type_name"),
      quantized?: Map.fetch!(tensor, "quantized?"),
      payload_bytes: Map.get(tensor, "payload_bytes", byte_size(Map.fetch!(tensor, "payload")))
    }
  end

  def compact_tensor_info(_tensor) do
    raise ArgumentError, "tensor is not a compact GGUF payload"
  end

  def fetch_compact_tensor(tensors, name) when is_map(tensors) and is_binary(name) do
    tensor = Map.fetch!(tensors, name)

    %{
      info: compact_tensor_info(tensor),
      payload: Map.fetch!(tensor, "payload")
    }
  end

  def dequantize_compact_tensor(%{info: %{type_name: "Q4_0", shape: shape}, payload: payload})
      when is_list(shape) and is_binary(payload) do
    count = Enum.product(shape)

    if rem(count, @q4_0_block_size) != 0 do
      raise ArgumentError, "Q4_0 tensor element count must be divisible by #{@q4_0_block_size}"
    end

    %{
      shape: shape,
      dtype: "f32",
      data: read_q4_0_blocks(payload, [])
    }
  end

  def dequantize_compact_tensor(%{info: %{type_name: type_name}}) do
    raise ArgumentError, "compact tensor type #{type_name} cannot be dequantized yet"
  end

  def dequantize_compact_tensor(_tensor) do
    raise ArgumentError, "tensor is not a compact GGUF payload"
  end

  def fetch_matrix(tensors, name) when is_map(tensors) and is_binary(name) do
    tensors
    |> Map.fetch!(name)
    |> Map.fetch!(:value)
  end

  def fetch_optional_matrix(tensors, name) when is_map(tensors) and is_binary(name) do
    case Map.fetch(tensors, name) do
      {:ok, tensor} -> Map.fetch!(tensor, :value)
      :error -> nil
    end
  end

  def layer_count(tensors) when is_map(tensors) do
    tensors
    |> Map.keys()
    |> Enum.filter(&is_binary/1)
    |> Enum.flat_map(fn name ->
      case Regex.run(~r/^blk\.(\d+)\./, name) do
        [_match, index] -> [String.to_integer(index)]
        nil -> []
      end
    end)
    |> Enum.max(fn -> -1 end)
    |> Kernel.+(1)
  end

  defp decode_tensor(name, %{"shape" => shape, "dtype" => dtype, "data" => data})
       when is_binary(name) and is_list(shape) and is_binary(dtype) and is_list(data) do
    validate_dtype!(name, dtype)
    validate_shape!(name, shape)
    validate_data_size!(name, shape, data)

    %{shape: shape, dtype: dtype, data: data, value: to_value(shape, data)}
  end

  defp decode_tensor(name, tensor) when is_binary(name) and is_map(tensor) do
    if compact_tensor?(tensor) do
      raise ArgumentError,
            "tensor #{name} is compact GGUF payload; dequantized tensor data is required"
    end

    raise ArgumentError, "tensor #{name} must contain shape, dtype, and data"
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

  defp to_value([size], data) when size == length(data), do: data

  defp to_value([rows, columns], data) do
    data
    |> Enum.chunk_every(columns)
    |> then(fn matrix ->
      if length(matrix) != rows do
        raise ArgumentError, "matrix row count does not match shape"
      end

      matrix
    end)
  end

  defp to_value(_shape, _data) do
    raise ArgumentError, "only rank-1 and rank-2 tensors are supported"
  end

  defp read_q4_0_blocks(<<>>, values), do: values |> Enum.reverse() |> List.flatten()

  defp read_q4_0_blocks(
         <<scale_bits::little-unsigned-integer-size(16),
           quantized::binary-size(div(@q4_0_block_size, 2)), rest::binary>>,
         values
       ) do
    scale = f16_to_float(scale_bits)
    block = quantized |> :binary.bin_to_list() |> Enum.flat_map(&q4_0_values(&1, scale))

    read_q4_0_blocks(rest, [block | values])
  end

  defp q4_0_values(byte, scale) do
    low = Bitwise.band(byte, 0x0F)
    high = byte |> Bitwise.bsr(4) |> Bitwise.band(0x0F)

    [(low - 8) * scale, (high - 8) * scale]
  end

  defp f16_to_float(bits) do
    sign = if Bitwise.band(bits, 0x8000) == 0, do: 1.0, else: -1.0
    exponent = bits |> Bitwise.bsr(10) |> Bitwise.band(0x1F)
    fraction = Bitwise.band(bits, 0x03FF)

    cond do
      exponent == 0 and fraction == 0 ->
        sign * 0.0

      exponent == 0 ->
        sign * :math.pow(2, -14) * (fraction / 1024)

      exponent == 31 and fraction == 0 ->
        if sign > 0.0, do: :positive_infinity, else: :negative_infinity

      exponent == 31 ->
        :nan

      true ->
        sign * :math.pow(2, exponent - 15) * (1 + fraction / 1024)
    end
  end
end
