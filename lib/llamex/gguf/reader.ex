defmodule Llamex.GGUF.Reader do
  @moduledoc """
  Reads GGUF header, metadata, and tensor directory.

  Supported tensor payloads are decoded into Llamex's flat named tensor schema.
  """

  @default_alignment 32
  @q4_0_block_size 32
  @q8_0_block_size 32

  defstruct [:version, :tensor_count, :metadata_count, :metadata, :tensors, :tensor_data_offset]

  @type t :: %__MODULE__{
          version: non_neg_integer(),
          tensor_count: non_neg_integer(),
          metadata_count: non_neg_integer(),
          metadata: map(),
          tensors: list(map()),
          tensor_data_offset: non_neg_integer()
        }

  def read_metadata(path) when is_binary(path) do
    path
    |> File.read!()
    |> read_binary()
  end

  def read_tensors(path) when is_binary(path) do
    binary = File.read!(path)

    read_tensor_data(read_binary(binary), binary)
  end

  def read_tensor_data(%__MODULE__{} = gguf, binary) when is_binary(binary) do
    Map.new(gguf.tensors, fn tensor ->
      {tensor.name, tensor_to_schema(gguf, tensor, binary)}
    end)
  end

  def read_binary(
        <<"GGUF", version::little-unsigned-integer-size(32),
          tensor_count::little-unsigned-integer-size(64),
          metadata_count::little-unsigned-integer-size(64), rest::binary>> = binary
      ) do
    total_size = byte_size(binary)
    {metadata, rest} = read_metadata_entries(rest, metadata_count, %{})
    {tensors, rest} = read_tensor_infos(rest, tensor_count, [])
    tensor_data_offset = total_size - byte_size(rest)

    tensor_data_offset =
      align_offset(
        tensor_data_offset,
        metadata_value(metadata, "general.alignment", @default_alignment)
      )

    %__MODULE__{
      version: version,
      tensor_count: tensor_count,
      metadata_count: metadata_count,
      metadata: metadata,
      tensors: Enum.reverse(tensors),
      tensor_data_offset: tensor_data_offset
    }
  end

  def read_binary(_binary), do: raise(ArgumentError, "not a GGUF file")

  defp read_metadata_entries(rest, 0, metadata), do: {metadata, rest}

  defp read_metadata_entries(binary, remaining, metadata) do
    {key, binary} = read_string(binary)
    {value_type, value, binary} = read_typed_value(binary)

    read_metadata_entries(
      binary,
      remaining - 1,
      Map.put(metadata, key, %{type: value_type, value: value})
    )
  end

  defp read_tensor_infos(rest, 0, tensors), do: {tensors, rest}

  defp read_tensor_infos(binary, remaining, tensors) do
    {name, binary} = read_string(binary)
    <<dimension_count::little-unsigned-integer-size(32), binary::binary>> = binary
    {dimensions, binary} = read_dimensions(binary, dimension_count, [])

    <<type::little-unsigned-integer-size(32), offset::little-unsigned-integer-size(64),
      binary::binary>> = binary

    tensor = %{
      name: name,
      dimensions: dimensions,
      type: type,
      offset: offset
    }

    read_tensor_infos(binary, remaining - 1, [tensor | tensors])
  end

  defp read_dimensions(binary, 0, dimensions), do: {Enum.reverse(dimensions), binary}

  defp read_dimensions(
         <<dimension::little-unsigned-integer-size(64), binary::binary>>,
         remaining,
         dimensions
       ) do
    read_dimensions(binary, remaining - 1, [dimension | dimensions])
  end

  defp read_typed_value(<<type::little-unsigned-integer-size(32), binary::binary>>) do
    {value, binary} = read_value(binary, type)
    {type_name(type), value, binary}
  end

  defp read_value(<<value::unsigned-integer-size(8), binary::binary>>, 0), do: {value, binary}
  defp read_value(<<value::signed-integer-size(8), binary::binary>>, 1), do: {value, binary}

  defp read_value(<<value::little-unsigned-integer-size(16), binary::binary>>, 2),
    do: {value, binary}

  defp read_value(<<value::little-signed-integer-size(16), binary::binary>>, 3),
    do: {value, binary}

  defp read_value(<<value::little-unsigned-integer-size(32), binary::binary>>, 4),
    do: {value, binary}

  defp read_value(<<value::little-signed-integer-size(32), binary::binary>>, 5),
    do: {value, binary}

  defp read_value(<<value::little-float-size(32), binary::binary>>, 6), do: {value, binary}
  defp read_value(<<0, binary::binary>>, 7), do: {false, binary}
  defp read_value(<<1, binary::binary>>, 7), do: {true, binary}
  defp read_value(binary, 8), do: read_string(binary)

  defp read_value(
         <<array_type::little-unsigned-integer-size(32), count::little-unsigned-integer-size(64),
           binary::binary>>,
         9
       ) do
    {values, binary} = read_array_values(binary, array_type, count, [])
    {%{type: type_name(array_type), values: Enum.reverse(values)}, binary}
  end

  defp read_value(<<value::little-unsigned-integer-size(64), binary::binary>>, 10),
    do: {value, binary}

  defp read_value(<<value::little-signed-integer-size(64), binary::binary>>, 11),
    do: {value, binary}

  defp read_value(<<value::little-float-size(64), binary::binary>>, 12), do: {value, binary}

  defp read_value(_binary, type),
    do: raise(ArgumentError, "unsupported GGUF metadata value type #{type}")

  defp read_array_values(binary, _array_type, 0, values), do: {values, binary}

  defp read_array_values(binary, array_type, remaining, values) do
    {value, binary} = read_value(binary, array_type)
    read_array_values(binary, array_type, remaining - 1, [value | values])
  end

  defp read_string(
         <<length::little-unsigned-integer-size(64), value::binary-size(length), rest::binary>>
       ) do
    {value, rest}
  end

  defp type_name(0), do: :uint8
  defp type_name(1), do: :int8
  defp type_name(2), do: :uint16
  defp type_name(3), do: :int16
  defp type_name(4), do: :uint32
  defp type_name(5), do: :int32
  defp type_name(6), do: :float32
  defp type_name(7), do: :bool
  defp type_name(8), do: :string
  defp type_name(9), do: :array
  defp type_name(10), do: :uint64
  defp type_name(11), do: :int64
  defp type_name(12), do: :float64
  defp type_name(type), do: {:unknown, type}

  defp align_offset(offset, alignment) when alignment > 0 do
    offset + rem(alignment - rem(offset, alignment), alignment)
  end

  defp metadata_value(metadata, key, default) do
    case Map.fetch(metadata, key) do
      {:ok, %{value: value}} -> value
      :error -> default
    end
  end

  defp tensor_to_schema(gguf, %{type: 0} = tensor, binary) do
    data =
      read_f32_tensor(
        binary,
        gguf.tensor_data_offset + tensor.offset,
        Enum.product(tensor.dimensions)
      )

    %{
      "shape" => schema_shape(tensor.dimensions),
      "dtype" => "f32",
      "data" => data
    }
  end

  defp tensor_to_schema(gguf, %{type: 1} = tensor, binary) do
    data =
      read_f16_tensor(
        binary,
        gguf.tensor_data_offset + tensor.offset,
        Enum.product(tensor.dimensions)
      )

    %{
      "shape" => schema_shape(tensor.dimensions),
      "dtype" => "f16",
      "data" => data
    }
  end

  defp tensor_to_schema(gguf, %{type: 2} = tensor, binary) do
    data =
      read_q4_0_tensor(
        binary,
        gguf.tensor_data_offset + tensor.offset,
        Enum.product(tensor.dimensions)
      )

    %{
      "shape" => schema_shape(tensor.dimensions),
      "dtype" => "f32",
      "data" => data
    }
  end

  defp tensor_to_schema(gguf, %{type: 8} = tensor, binary) do
    data =
      read_q8_0_tensor(
        binary,
        gguf.tensor_data_offset + tensor.offset,
        Enum.product(tensor.dimensions)
      )

    %{
      "shape" => schema_shape(tensor.dimensions),
      "dtype" => "f32",
      "data" => data
    }
  end

  defp tensor_to_schema(_gguf, tensor, _binary) do
    raise ArgumentError, "unsupported GGUF tensor type #{tensor.type} for #{tensor.name}"
  end

  defp read_f32_tensor(binary, offset, count) do
    byte_size = count * 4
    <<_prefix::binary-size(offset), tensor_data::binary-size(byte_size), _rest::binary>> = binary
    read_f32_values(tensor_data, [])
  end

  defp read_f32_values(<<>>, values), do: Enum.reverse(values)

  defp read_f32_values(<<value::little-float-size(32), rest::binary>>, values) do
    read_f32_values(rest, [value | values])
  end

  defp read_f16_tensor(binary, offset, count) do
    byte_size = count * 2
    <<_prefix::binary-size(offset), tensor_data::binary-size(byte_size), _rest::binary>> = binary
    read_f16_values(tensor_data, [])
  end

  defp read_f16_values(<<>>, values), do: Enum.reverse(values)

  defp read_f16_values(<<bits::little-unsigned-integer-size(16), rest::binary>>, values) do
    read_f16_values(rest, [f16_to_float(bits) | values])
  end

  defp read_q4_0_tensor(_binary, _offset, count) when rem(count, @q4_0_block_size) != 0 do
    raise ArgumentError, "Q4_0 tensor element count must be divisible by #{@q4_0_block_size}"
  end

  defp read_q4_0_tensor(binary, offset, count) do
    byte_size = div(count, @q4_0_block_size) * (2 + div(@q4_0_block_size, 2))
    <<_prefix::binary-size(offset), tensor_data::binary-size(byte_size), _rest::binary>> = binary
    read_q4_0_blocks(tensor_data, [])
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

  defp read_q8_0_tensor(_binary, _offset, count) when rem(count, @q8_0_block_size) != 0 do
    raise ArgumentError, "Q8_0 tensor element count must be divisible by #{@q8_0_block_size}"
  end

  defp read_q8_0_tensor(binary, offset, count) do
    byte_size = div(count, @q8_0_block_size) * (2 + @q8_0_block_size)
    <<_prefix::binary-size(offset), tensor_data::binary-size(byte_size), _rest::binary>> = binary
    read_q8_0_blocks(tensor_data, [])
  end

  defp read_q8_0_blocks(<<>>, values), do: values |> Enum.reverse() |> List.flatten()

  defp read_q8_0_blocks(
         <<scale_bits::little-unsigned-integer-size(16), quantized::binary-size(@q8_0_block_size),
           rest::binary>>,
         values
       ) do
    scale = f16_to_float(scale_bits)

    block =
      quantized
      |> :binary.bin_to_list()
      |> Enum.map(&signed_i8/1)
      |> Enum.map(&(&1 * scale))

    read_q8_0_blocks(rest, [block | values])
  end

  defp signed_i8(value) when value < 128, do: value
  defp signed_i8(value), do: value - 256

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

  defp schema_shape([_size] = dimensions), do: dimensions
  defp schema_shape(dimensions), do: Enum.reverse(dimensions)
end
