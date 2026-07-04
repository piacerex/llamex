defmodule Llamex.GGUF.Diagnostic do
  @moduledoc """
  Diagnostics for GGUF model compatibility.
  """

  @supported_tensor_types %{
    0 => "F32",
    1 => "F16",
    2 => "Q4_0",
    3 => "Q4_1",
    6 => "Q5_0",
    7 => "Q5_1",
    8 => "Q8_0",
    9 => "Q8_1",
    12 => "Q4_K",
    13 => "Q5_K",
    14 => "Q6_K",
    15 => "Q8_K"
  }

  def inspect_file(path) when is_binary(path) do
    path
    |> File.read!()
    |> inspect_binary()
  end

  def inspect_binary(binary) when is_binary(binary) do
    gguf = Llamex.GGUF.Reader.read_binary(binary)

    %{
      version: gguf.version,
      tensor_count: gguf.tensor_count,
      metadata_count: gguf.metadata_count,
      architecture: metadata_value(gguf.metadata, "general.architecture"),
      tokenizer_token_count: tokenizer_token_count(gguf.metadata),
      supported_tensor_types: supported_tensor_types(gguf.tensors),
      unsupported_tensor_types: unsupported_tensor_types(gguf.tensors),
      unsupported_tensors: unsupported_tensors(gguf.tensors)
    }
  end

  def format(%{} = diagnostic) do
    [
      "GGUF v#{diagnostic.version}",
      "architecture: #{diagnostic.architecture || "unknown"}",
      "metadata: #{diagnostic.metadata_count}",
      "tensors: #{diagnostic.tensor_count}",
      "tokenizer tokens: #{diagnostic.tokenizer_token_count || "unknown"}",
      "supported tensor types: #{format_type_counts(diagnostic.supported_tensor_types)}",
      "unsupported tensor types: #{format_type_counts(diagnostic.unsupported_tensor_types)}",
      format_unsupported_tensors(diagnostic.unsupported_tensors)
    ]
    |> Enum.reject(&(&1 == ""))
    |> Enum.join("\n")
  end

  defp supported_tensor_types(tensors) do
    tensors
    |> Enum.filter(&supported_tensor_type?/1)
    |> type_counts()
  end

  defp unsupported_tensor_types(tensors) do
    tensors
    |> Enum.reject(&supported_tensor_type?/1)
    |> type_counts()
  end

  defp unsupported_tensors(tensors) do
    tensors
    |> Enum.reject(&supported_tensor_type?/1)
    |> Enum.map(fn tensor ->
      %{
        name: tensor.name,
        type: tensor.type,
        dimensions: tensor.dimensions
      }
    end)
  end

  defp supported_tensor_type?(%{type: type}), do: Map.has_key?(@supported_tensor_types, type)

  defp type_counts(tensors) do
    tensors
    |> Enum.frequencies_by(& &1.type)
    |> Map.new(fn {type, count} -> {tensor_type_name(type), count} end)
  end

  defp tensor_type_name(type), do: Map.get(@supported_tensor_types, type, "type_#{type}")

  defp tokenizer_token_count(metadata) do
    case metadata_value(metadata, "tokenizer.ggml.tokens") do
      %{values: values} -> length(values)
      _other -> nil
    end
  end

  defp metadata_value(metadata, key) do
    case Map.fetch(metadata, key) do
      {:ok, %{value: value}} -> value
      :error -> nil
    end
  end

  defp format_type_counts(counts) when map_size(counts) == 0, do: "none"

  defp format_type_counts(counts) do
    counts
    |> Enum.sort()
    |> Enum.map(fn {type, count} -> "#{type}=#{count}" end)
    |> Enum.join(", ")
  end

  defp format_unsupported_tensors([]), do: ""

  defp format_unsupported_tensors(tensors) do
    tensors =
      tensors
      |> Enum.map(fn tensor ->
        "- #{tensor.name}: type_#{tensor.type} #{inspect(tensor.dimensions)}"
      end)
      |> Enum.join("\n")

    "unsupported tensors:\n" <> tensors
  end
end
