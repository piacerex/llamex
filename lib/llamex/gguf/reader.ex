defmodule Llamex.GGUF.Reader do
  @moduledoc """
  Reads GGUF header, metadata, and tensor directory.

  Supported tensor payloads are decoded into Llamex's flat named tensor schema.
  Compact tensor payloads can also be read without eager F32 dequantization for
  diagnostics and future backends that can consume quantized GGUF blocks.
  """

  @default_alignment 32
  @q4_0_block_size 32
  @q4_1_block_size 32
  @q2_k_block_size 256
  @q3_k_block_size 256
  @q4_k_block_size 256
  @q4_k_scale_size 12
  @q5_0_block_size 32
  @q5_1_block_size 32
  @q5_k_block_size 256
  @q8_0_block_size 32
  @q8_1_block_size 32
  @q6_k_block_size 256
  @q8_k_block_size 256

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

  def read_compact_tensor_data(%__MODULE__{} = gguf, binary) when is_binary(binary) do
    Map.new(gguf.tensors, fn tensor ->
      {tensor.name, tensor_to_compact_schema(gguf, tensor, binary)}
    end)
  end

  def read_tensor_data(%__MODULE__{} = gguf, binary) when is_binary(binary) do
    gguf.tensors
    |> Task.async_stream(
      fn tensor ->
        try do
          {:ok, {tensor.name, tensor_to_schema(gguf, tensor, binary)}}
        rescue
          exception -> {:error, exception}
        end
      end,
      ordered: false,
      timeout: :infinity,
      max_concurrency: System.schedulers_online()
    )
    |> Map.new(fn
      {:ok, {:ok, tensor}} -> tensor
      {:ok, {:error, exception}} -> raise exception
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

  defp tensor_to_schema(gguf, %{type: 3} = tensor, binary) do
    data =
      read_q4_1_tensor(
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

  defp tensor_to_schema(gguf, %{type: 6} = tensor, binary) do
    data =
      read_q5_0_tensor(
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

  defp tensor_to_schema(gguf, %{type: 7} = tensor, binary) do
    data =
      read_q5_1_tensor(
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

  defp tensor_to_schema(gguf, %{type: 9} = tensor, binary) do
    data =
      read_q8_1_tensor(
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

  defp tensor_to_schema(gguf, %{type: 10} = tensor, binary) do
    data =
      read_q2_k_tensor(
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

  defp tensor_to_schema(gguf, %{type: 11} = tensor, binary) do
    data =
      read_q3_k_tensor(
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

  defp tensor_to_schema(gguf, %{type: 12} = tensor, binary) do
    data =
      read_q4_k_tensor(
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

  defp tensor_to_schema(gguf, %{type: 13} = tensor, binary) do
    data =
      read_q5_k_tensor(
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

  defp tensor_to_schema(gguf, %{type: 14} = tensor, binary) do
    data =
      read_q6_k_tensor(
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

  defp tensor_to_schema(gguf, %{type: 15} = tensor, binary) do
    data =
      read_q8_k_tensor(
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

  defp tensor_to_schema(gguf, %{type: 30} = tensor, binary) do
    data =
      read_bf16_tensor(
        binary,
        gguf.tensor_data_offset + tensor.offset,
        Enum.product(tensor.dimensions)
      )

    %{
      "shape" => schema_shape(tensor.dimensions),
      "dtype" => "bf16",
      "data" => data
    }
  end

  defp tensor_to_schema(_gguf, tensor, _binary) do
    raise ArgumentError, "unsupported GGUF tensor type #{tensor.type} for #{tensor.name}"
  end

  defp tensor_to_compact_schema(gguf, tensor, binary) do
    byte_size = tensor_payload_bytes!(tensor)
    offset = gguf.tensor_data_offset + tensor.offset
    <<_prefix::binary-size(offset), payload::binary-size(byte_size), _rest::binary>> = binary

    %{
      "shape" => schema_shape(tensor.dimensions),
      "dtype" => tensor_dtype(tensor.type),
      "type" => tensor.type,
      "type_name" => tensor_type_name(tensor.type),
      "quantized?" => quantized_tensor_type?(tensor.type),
      "payload" => payload,
      "payload_bytes" => byte_size
    }
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

  defp tensor_payload_bytes!(%{type: 0, dimensions: dimensions}),
    do: element_count(dimensions) * 4

  defp tensor_payload_bytes!(%{type: 1, dimensions: dimensions}),
    do: element_count(dimensions) * 2

  defp tensor_payload_bytes!(%{type: 30, dimensions: dimensions}),
    do: element_count(dimensions) * 2

  defp tensor_payload_bytes!(%{type: type, dimensions: dimensions})
       when type in [2, 3, 6, 7, 8, 9] do
    {block_size, bytes_per_block} =
      case type do
        2 -> {@q4_0_block_size, 2 + div(@q4_0_block_size, 2)}
        3 -> {@q4_1_block_size, 4 + div(@q4_1_block_size, 2)}
        6 -> {@q5_0_block_size, 2 + 4 + div(@q5_0_block_size, 2)}
        7 -> {@q5_1_block_size, 4 + 4 + div(@q5_1_block_size, 2)}
        8 -> {@q8_0_block_size, 2 + @q8_0_block_size}
        9 -> {@q8_1_block_size, 4 + @q8_1_block_size}
      end

    block_payload_bytes!(type, dimensions, block_size, bytes_per_block)
  end

  defp tensor_payload_bytes!(%{type: type, dimensions: dimensions})
       when type in [10, 11, 12, 13, 14, 15] do
    {block_size, bytes_per_block} =
      case type do
        10 -> {@q2_k_block_size, 16 + 64 + 4}
        11 -> {@q3_k_block_size, 32 + 64 + 12 + 2}
        12 -> {@q4_k_block_size, 4 + @q4_k_scale_size + div(@q4_k_block_size, 2)}
        13 -> {@q5_k_block_size, 4 + 12 + div(@q5_k_block_size, 8) + div(@q5_k_block_size, 2)}
        14 -> {@q6_k_block_size, 128 + 64 + 16 + 2}
        15 -> {@q8_k_block_size, 4 + @q8_k_block_size + 32}
      end

    block_payload_bytes!(type, dimensions, block_size, bytes_per_block)
  end

  defp tensor_payload_bytes!(tensor) do
    raise ArgumentError, "unsupported GGUF tensor type #{tensor.type} for #{tensor.name}"
  end

  defp block_payload_bytes!(type, dimensions, block_size, bytes_per_block) do
    count = element_count(dimensions)

    if rem(count, block_size) == 0 do
      div(count, block_size) * bytes_per_block
    else
      raise ArgumentError,
            "#{tensor_type_name(type)} tensor element count must be divisible by #{block_size}"
    end
  end

  defp tensor_dtype(0), do: "f32"
  defp tensor_dtype(1), do: "f16"
  defp tensor_dtype(30), do: "bf16"
  defp tensor_dtype(_type), do: "quantized"

  defp quantized_tensor_type?(type), do: type not in [0, 1, 30]

  defp tensor_type_name(0), do: "F32"
  defp tensor_type_name(1), do: "F16"
  defp tensor_type_name(2), do: "Q4_0"
  defp tensor_type_name(3), do: "Q4_1"
  defp tensor_type_name(6), do: "Q5_0"
  defp tensor_type_name(7), do: "Q5_1"
  defp tensor_type_name(8), do: "Q8_0"
  defp tensor_type_name(9), do: "Q8_1"
  defp tensor_type_name(10), do: "Q2_K"
  defp tensor_type_name(11), do: "Q3_K"
  defp tensor_type_name(12), do: "Q4_K"
  defp tensor_type_name(13), do: "Q5_K"
  defp tensor_type_name(14), do: "Q6_K"
  defp tensor_type_name(15), do: "Q8_K"
  defp tensor_type_name(30), do: "BF16"
  defp tensor_type_name(type), do: "type_#{type}"

  defp element_count(dimensions), do: Enum.product(dimensions)

  defp read_f16_tensor(binary, offset, count) do
    byte_size = count * 2
    <<_prefix::binary-size(offset), tensor_data::binary-size(byte_size), _rest::binary>> = binary
    read_f16_values(tensor_data, [])
  end

  defp read_f16_values(<<>>, values), do: Enum.reverse(values)

  defp read_f16_values(<<bits::little-unsigned-integer-size(16), rest::binary>>, values) do
    read_f16_values(rest, [f16_to_float(bits) | values])
  end

  defp read_bf16_tensor(binary, offset, count) do
    byte_size = count * 2
    <<_prefix::binary-size(offset), tensor_data::binary-size(byte_size), _rest::binary>> = binary
    read_bf16_values(tensor_data, [])
  end

  defp read_bf16_values(<<>>, values), do: Enum.reverse(values)

  defp read_bf16_values(<<bits::little-unsigned-integer-size(16), rest::binary>>, values) do
    read_bf16_values(rest, [bf16_to_float(bits) | values])
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

  defp read_q4_1_tensor(_binary, _offset, count) when rem(count, @q4_1_block_size) != 0 do
    raise ArgumentError, "Q4_1 tensor element count must be divisible by #{@q4_1_block_size}"
  end

  defp read_q4_1_tensor(binary, offset, count) do
    byte_size = div(count, @q4_1_block_size) * (4 + div(@q4_1_block_size, 2))
    <<_prefix::binary-size(offset), tensor_data::binary-size(byte_size), _rest::binary>> = binary
    read_q4_1_blocks(tensor_data, [])
  end

  defp read_q4_1_blocks(<<>>, values), do: values |> Enum.reverse() |> List.flatten()

  defp read_q4_1_blocks(
         <<scale_bits::little-unsigned-integer-size(16),
           min_bits::little-unsigned-integer-size(16),
           quantized::binary-size(div(@q4_1_block_size, 2)), rest::binary>>,
         values
       ) do
    scale = f16_to_float(scale_bits)
    minimum = f16_to_float(min_bits)
    block = quantized |> :binary.bin_to_list() |> Enum.flat_map(&q4_1_values(&1, scale, minimum))

    read_q4_1_blocks(rest, [block | values])
  end

  defp q4_1_values(byte, scale, minimum) do
    low = Bitwise.band(byte, 0x0F)
    high = byte |> Bitwise.bsr(4) |> Bitwise.band(0x0F)

    [low * scale + minimum, high * scale + minimum]
  end

  defp read_q5_0_tensor(_binary, _offset, count) when rem(count, @q5_0_block_size) != 0 do
    raise ArgumentError, "Q5_0 tensor element count must be divisible by #{@q5_0_block_size}"
  end

  defp read_q5_0_tensor(binary, offset, count) do
    byte_size = div(count, @q5_0_block_size) * (2 + 4 + div(@q5_0_block_size, 2))
    <<_prefix::binary-size(offset), tensor_data::binary-size(byte_size), _rest::binary>> = binary
    read_q5_0_blocks(tensor_data, [])
  end

  defp read_q5_0_blocks(<<>>, values), do: values |> Enum.reverse() |> List.flatten()

  defp read_q5_0_blocks(
         <<scale_bits::little-unsigned-integer-size(16),
           high_bits::little-unsigned-integer-size(32),
           quantized::binary-size(div(@q5_0_block_size, 2)), rest::binary>>,
         values
       ) do
    scale = f16_to_float(scale_bits)

    block =
      quantized
      |> :binary.bin_to_list()
      |> Enum.with_index()
      |> Enum.flat_map(fn {byte, byte_index} ->
        q5_0_values(byte, scale, high_bits, byte_index * 2)
      end)

    read_q5_0_blocks(rest, [block | values])
  end

  defp q5_0_values(byte, scale, high_bits, low_index) do
    low = Bitwise.band(byte, 0x0F)
    high = byte |> Bitwise.bsr(4) |> Bitwise.band(0x0F)

    [
      q5_0_value(low, scale, high_bits, low_index),
      q5_0_value(high, scale, high_bits, low_index + 1)
    ]
  end

  defp q5_0_value(low_bits, scale, high_bits, index) do
    high_bit = high_bits |> Bitwise.bsr(index) |> Bitwise.band(0x01) |> Bitwise.bsl(4)

    (low_bits + high_bit - 16) * scale
  end

  defp read_q5_1_tensor(_binary, _offset, count) when rem(count, @q5_1_block_size) != 0 do
    raise ArgumentError, "Q5_1 tensor element count must be divisible by #{@q5_1_block_size}"
  end

  defp read_q5_1_tensor(binary, offset, count) do
    byte_size = div(count, @q5_1_block_size) * (4 + 4 + div(@q5_1_block_size, 2))
    <<_prefix::binary-size(offset), tensor_data::binary-size(byte_size), _rest::binary>> = binary
    read_q5_1_blocks(tensor_data, [])
  end

  defp read_q5_1_blocks(<<>>, values), do: values |> Enum.reverse() |> List.flatten()

  defp read_q5_1_blocks(
         <<scale_bits::little-unsigned-integer-size(16),
           min_bits::little-unsigned-integer-size(16),
           high_bits::little-unsigned-integer-size(32),
           quantized::binary-size(div(@q5_1_block_size, 2)), rest::binary>>,
         values
       ) do
    scale = f16_to_float(scale_bits)
    minimum = f16_to_float(min_bits)

    block =
      quantized
      |> :binary.bin_to_list()
      |> Enum.with_index()
      |> Enum.flat_map(fn {byte, byte_index} ->
        q5_1_values(byte, scale, minimum, high_bits, byte_index * 2)
      end)

    read_q5_1_blocks(rest, [block | values])
  end

  defp q5_1_values(byte, scale, minimum, high_bits, low_index) do
    low = Bitwise.band(byte, 0x0F)
    high = byte |> Bitwise.bsr(4) |> Bitwise.band(0x0F)

    [
      q5_1_value(low, scale, minimum, high_bits, low_index),
      q5_1_value(high, scale, minimum, high_bits, low_index + 1)
    ]
  end

  defp q5_1_value(low_bits, scale, minimum, high_bits, index) do
    high_bit = high_bits |> Bitwise.bsr(index) |> Bitwise.band(0x01) |> Bitwise.bsl(4)

    (low_bits + high_bit) * scale + minimum
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

  defp read_q8_1_tensor(_binary, _offset, count) when rem(count, @q8_1_block_size) != 0 do
    raise ArgumentError, "Q8_1 tensor element count must be divisible by #{@q8_1_block_size}"
  end

  defp read_q8_1_tensor(binary, offset, count) do
    byte_size = div(count, @q8_1_block_size) * (4 + @q8_1_block_size)
    <<_prefix::binary-size(offset), tensor_data::binary-size(byte_size), _rest::binary>> = binary
    read_q8_1_blocks(tensor_data, [])
  end

  defp read_q8_1_blocks(<<>>, values), do: values |> Enum.reverse() |> List.flatten()

  defp read_q8_1_blocks(
         <<scale_bits::little-unsigned-integer-size(16),
           _sum_bits::little-unsigned-integer-size(16), quantized::binary-size(@q8_1_block_size),
           rest::binary>>,
         values
       ) do
    scale = f16_to_float(scale_bits)

    block =
      quantized
      |> :binary.bin_to_list()
      |> Enum.map(&signed_i8/1)
      |> Enum.map(&(&1 * scale))

    read_q8_1_blocks(rest, [block | values])
  end

  defp read_q2_k_tensor(_binary, _offset, count) when rem(count, @q2_k_block_size) != 0 do
    raise ArgumentError, "Q2_K tensor element count must be divisible by #{@q2_k_block_size}"
  end

  defp read_q2_k_tensor(binary, offset, count) do
    byte_size = div(count, @q2_k_block_size) * (16 + 64 + 4)
    <<_prefix::binary-size(offset), tensor_data::binary-size(byte_size), _rest::binary>> = binary
    read_q2_k_blocks(tensor_data, [])
  end

  defp read_q2_k_blocks(<<>>, values), do: values |> Enum.reverse() |> List.flatten()

  defp read_q2_k_blocks(
         <<scales::binary-size(16), quantized::binary-size(64),
           scale_bits::little-unsigned-integer-size(16),
           min_bits::little-unsigned-integer-size(16), rest::binary>>,
         values
       ) do
    scale = f16_to_float(scale_bits)
    minimum = f16_to_float(min_bits)

    block =
      0..(@q2_k_block_size - 1)
      |> Enum.map(fn index ->
        q2_k_value(quantized, scales, scale, minimum, index)
      end)

    read_q2_k_blocks(rest, [block | values])
  end

  defp q2_k_value(quantized, scales, scale, minimum, index) do
    super_group_index = div(index, 128)
    super_group_offset = rem(index, 128)
    pair_index = div(super_group_offset, 32)
    pair_offset = rem(super_group_offset, 32)
    scale_index = super_group_index * 8 + pair_index * 2 + if(pair_offset < 16, do: 0, else: 1)
    scale_min = :binary.at(scales, scale_index)
    block_scale = Bitwise.band(scale_min, 0x0F)
    block_minimum = Bitwise.bsr(scale_min, 4)

    quant_index =
      super_group_index * 32 + rem(pair_offset, 16) + if(pair_offset < 16, do: 0, else: 16)

    shift = pair_index * 2

    quantized
    |> :binary.at(quant_index)
    |> Bitwise.bsr(shift)
    |> Bitwise.band(0x03)
    |> then(&(&1 * block_scale * scale - block_minimum * minimum))
  end

  defp read_q3_k_tensor(_binary, _offset, count) when rem(count, @q3_k_block_size) != 0 do
    raise ArgumentError, "Q3_K tensor element count must be divisible by #{@q3_k_block_size}"
  end

  defp read_q3_k_tensor(binary, offset, count) do
    byte_size = div(count, @q3_k_block_size) * (32 + 64 + 12 + 2)
    <<_prefix::binary-size(offset), tensor_data::binary-size(byte_size), _rest::binary>> = binary
    read_q3_k_blocks(tensor_data, [])
  end

  defp read_q3_k_blocks(<<>>, values), do: values |> Enum.reverse() |> List.flatten()

  defp read_q3_k_blocks(
         <<high_mask::binary-size(32), quantized::binary-size(64), scales::binary-size(12),
           scale_bits::little-unsigned-integer-size(16), rest::binary>>,
         values
       ) do
    scale = f16_to_float(scale_bits)
    scales = q3_k_scales(scales)

    block =
      0..(@q3_k_block_size - 1)
      |> Enum.map(fn index ->
        q3_k_value(quantized, high_mask, scales, scale, index)
      end)

    read_q3_k_blocks(rest, [block | values])
  end

  defp q3_k_value(quantized, high_mask, scales, scale, index) do
    super_group_index = div(index, 128)
    super_group_offset = rem(index, 128)
    pair_index = div(super_group_offset, 32)
    pair_offset = rem(super_group_offset, 32)
    scale_index = super_group_index * 8 + pair_index * 2 + if(pair_offset < 16, do: 0, else: 1)

    quant_index =
      super_group_index * 32 + rem(pair_offset, 16) + if(pair_offset < 16, do: 0, else: 16)

    shift = pair_index * 2

    quant =
      quantized
      |> :binary.at(quant_index)
      |> Bitwise.bsr(shift)
      |> Bitwise.band(0x03)

    sign_offset =
      if Bitwise.band(:binary.at(high_mask, rem(index, 32)), Bitwise.bsl(1, div(index, 32))) == 0,
        do: 4,
        else: 0

    scale * q3_k_scale(scales, scale_index) * (quant - sign_offset)
  end

  defp q3_k_scale(scales, index), do: Enum.at(scales, index)

  defp q3_k_scales(scales) do
    aux0 = u32_at(scales, 0)
    aux1 = u32_at(scales, 4)
    aux2 = u32_at(scales, 8)

    [
      Bitwise.bor(Bitwise.band(aux0, 0x0F0F0F0F), Bitwise.bsl(Bitwise.band(aux2, 0x03030303), 4)),
      Bitwise.bor(
        Bitwise.band(aux1, 0x0F0F0F0F),
        Bitwise.bsl(Bitwise.band(Bitwise.bsr(aux2, 2), 0x03030303), 4)
      ),
      Bitwise.bor(
        Bitwise.band(Bitwise.bsr(aux0, 4), 0x0F0F0F0F),
        Bitwise.bsl(Bitwise.band(Bitwise.bsr(aux2, 4), 0x03030303), 4)
      ),
      Bitwise.bor(
        Bitwise.band(Bitwise.bsr(aux1, 4), 0x0F0F0F0F),
        Bitwise.bsl(Bitwise.band(Bitwise.bsr(aux2, 6), 0x03030303), 4)
      )
    ]
    |> Enum.flat_map(&u32_bytes/1)
    |> Enum.map(&signed_i8/1)
    |> Enum.map(&(&1 - 32))
  end

  defp u32_at(binary, offset) do
    <<_prefix::binary-size(offset), value::little-unsigned-integer-size(32), _rest::binary>> =
      binary

    value
  end

  defp u32_bytes(value) do
    [
      Bitwise.band(value, 0xFF),
      Bitwise.band(Bitwise.bsr(value, 8), 0xFF),
      Bitwise.band(Bitwise.bsr(value, 16), 0xFF),
      Bitwise.band(Bitwise.bsr(value, 24), 0xFF)
    ]
  end

  defp read_q4_k_tensor(_binary, _offset, count) when rem(count, @q4_k_block_size) != 0 do
    raise ArgumentError, "Q4_K tensor element count must be divisible by #{@q4_k_block_size}"
  end

  defp read_q4_k_tensor(binary, offset, count) do
    byte_size =
      div(count, @q4_k_block_size) *
        (4 + @q4_k_scale_size + div(@q4_k_block_size, 2))

    <<_prefix::binary-size(offset), tensor_data::binary-size(byte_size), _rest::binary>> = binary
    read_q4_k_blocks(tensor_data, [])
  end

  defp read_q4_k_blocks(<<>>, values), do: values |> Enum.reverse() |> List.flatten()

  defp read_q4_k_blocks(
         <<scale_bits::little-unsigned-integer-size(16),
           min_bits::little-unsigned-integer-size(16), scales::binary-size(@q4_k_scale_size),
           quantized::binary-size(div(@q4_k_block_size, 2)), rest::binary>>,
         values
       ) do
    scale = f16_to_float(scale_bits)
    minimum = f16_to_float(min_bits)

    block =
      0..(@q4_k_block_size - 1)
      |> Enum.map(fn index ->
        q4_k_value(quantized, scales, scale, minimum, index)
      end)

    read_q4_k_blocks(rest, [block | values])
  end

  defp q4_k_value(quantized, scales, scale, minimum, index) do
    group_index = div(index, 64)
    group_offset = rem(index, 64)
    scale_index = group_index * 2 + if(group_offset < 32, do: 0, else: 1)
    quant_byte = :binary.at(quantized, group_index * 32 + rem(group_offset, 32))

    quant =
      if group_offset < 32 do
        Bitwise.band(quant_byte, 0x0F)
      else
        Bitwise.bsr(quant_byte, 4)
      end

    {block_scale, block_minimum} = q4_k_scale_min(scales, scale_index)

    scale * block_scale * quant - minimum * block_minimum
  end

  defp q4_k_scale_min(scales, index) when index < 4 do
    {
      scales |> :binary.at(index) |> Bitwise.band(0x3F),
      scales |> :binary.at(index + 4) |> Bitwise.band(0x3F)
    }
  end

  defp q4_k_scale_min(scales, index) do
    scale =
      Bitwise.bor(
        scales |> :binary.at(index + 4) |> Bitwise.band(0x0F),
        scales |> :binary.at(index - 4) |> Bitwise.bsr(6) |> Bitwise.bsl(4)
      )

    minimum =
      Bitwise.bor(
        scales |> :binary.at(index + 4) |> Bitwise.bsr(4),
        scales |> :binary.at(index) |> Bitwise.bsr(6) |> Bitwise.bsl(4)
      )

    {scale, minimum}
  end

  defp read_q5_k_tensor(_binary, _offset, count) when rem(count, @q5_k_block_size) != 0 do
    raise ArgumentError, "Q5_K tensor element count must be divisible by #{@q5_k_block_size}"
  end

  defp read_q5_k_tensor(binary, offset, count) do
    byte_size =
      div(count, @q5_k_block_size) *
        (4 + @q4_k_scale_size + div(@q5_k_block_size, 8) + div(@q5_k_block_size, 2))

    <<_prefix::binary-size(offset), tensor_data::binary-size(byte_size), _rest::binary>> = binary
    read_q5_k_blocks(tensor_data, [])
  end

  defp read_q5_k_blocks(<<>>, values), do: values |> Enum.reverse() |> List.flatten()

  defp read_q5_k_blocks(
         <<scale_bits::little-unsigned-integer-size(16),
           min_bits::little-unsigned-integer-size(16), scales::binary-size(@q4_k_scale_size),
           high_bits::binary-size(div(@q5_k_block_size, 8)),
           quantized::binary-size(div(@q5_k_block_size, 2)), rest::binary>>,
         values
       ) do
    scale = f16_to_float(scale_bits)
    minimum = f16_to_float(min_bits)

    block =
      0..(@q5_k_block_size - 1)
      |> Enum.map(fn index ->
        q5_k_value(quantized, high_bits, scales, scale, minimum, index)
      end)

    read_q5_k_blocks(rest, [block | values])
  end

  defp q5_k_value(quantized, high_bits, scales, scale, minimum, index) do
    group_index = div(index, 64)
    group_offset = rem(index, 64)
    scale_index = group_index * 2 + if(group_offset < 32, do: 0, else: 1)
    quant_byte = :binary.at(quantized, group_index * 32 + rem(group_offset, 32))
    high_mask = Bitwise.bsl(if(group_offset < 32, do: 1, else: 2), group_index * 2)

    high_bit =
      if Bitwise.band(:binary.at(high_bits, rem(group_offset, 32)), high_mask) == 0,
        do: 0,
        else: 16

    quant =
      if group_offset < 32 do
        Bitwise.band(quant_byte, 0x0F)
      else
        Bitwise.bsr(quant_byte, 4)
      end

    {block_scale, block_minimum} = q4_k_scale_min(scales, scale_index)

    scale * block_scale * (quant + high_bit) - minimum * block_minimum
  end

  defp read_q6_k_tensor(_binary, _offset, count) when rem(count, @q6_k_block_size) != 0 do
    raise ArgumentError, "Q6_K tensor element count must be divisible by #{@q6_k_block_size}"
  end

  defp read_q6_k_tensor(binary, offset, count) do
    byte_size = div(count, @q6_k_block_size) * (128 + 64 + 16 + 2)
    <<_prefix::binary-size(offset), tensor_data::binary-size(byte_size), _rest::binary>> = binary
    read_q6_k_blocks(tensor_data, [])
  end

  defp read_q6_k_blocks(<<>>, values), do: values |> Enum.reverse() |> List.flatten()

  defp read_q6_k_blocks(
         <<ql::binary-size(128), qh::binary-size(64), scales::binary-size(16),
           scale_bits::little-unsigned-integer-size(16), rest::binary>>,
         values
       ) do
    scale = f16_to_float(scale_bits)

    block =
      0..(@q6_k_block_size - 1)
      |> Enum.map(fn index ->
        q6_k_value(ql, qh, scales, scale, index)
      end)

    read_q6_k_blocks(rest, [block | values])
  end

  defp q6_k_value(ql, qh, scales, scale, index) do
    group_index = div(index, 128)
    group_offset = rem(index, 128)
    quadrant = div(group_offset, 32)
    offset = rem(group_offset, 32)

    ql_index = group_index * 64 + offset + if(quadrant in [1, 3], do: 32, else: 0)

    low_bits =
      ql
      |> :binary.at(ql_index)
      |> then(fn byte ->
        if quadrant in [0, 1], do: Bitwise.band(byte, 0x0F), else: Bitwise.bsr(byte, 4)
      end)

    high_bits =
      qh
      |> :binary.at(group_index * 32 + offset)
      |> Bitwise.bsr(quadrant * 2)
      |> Bitwise.band(0x03)
      |> Bitwise.bsl(4)

    quant = low_bits + high_bits - 32

    block_scale =
      scales |> :binary.at(group_index * 8 + div(offset, 16) + quadrant * 2) |> signed_i8()

    quant * block_scale * scale
  end

  defp read_q8_k_tensor(_binary, _offset, count) when rem(count, @q8_k_block_size) != 0 do
    raise ArgumentError, "Q8_K tensor element count must be divisible by #{@q8_k_block_size}"
  end

  defp read_q8_k_tensor(binary, offset, count) do
    byte_size = div(count, @q8_k_block_size) * (4 + @q8_k_block_size + 32)
    <<_prefix::binary-size(offset), tensor_data::binary-size(byte_size), _rest::binary>> = binary
    read_q8_k_blocks(tensor_data, [])
  end

  defp read_q8_k_blocks(<<>>, values), do: values |> Enum.reverse() |> List.flatten()

  defp read_q8_k_blocks(
         <<scale::little-float-size(32), quantized::binary-size(@q8_k_block_size),
           _sums::binary-size(32), rest::binary>>,
         values
       ) do
    block =
      quantized
      |> :binary.bin_to_list()
      |> Enum.map(&signed_i8/1)
      |> Enum.map(&(&1 * scale))

    read_q8_k_blocks(rest, [block | values])
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

  defp bf16_to_float(bits) do
    <<value::little-float-size(32)>> =
      <<0::little-unsigned-integer-size(16), bits::little-unsigned-integer-size(16)>>

    value
  end

  defp schema_shape([_size] = dimensions), do: dimensions
  defp schema_shape(dimensions), do: Enum.reverse(dimensions)
end
