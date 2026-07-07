defmodule Llamex.GGUF.Diagnostic do
  @moduledoc """
  Diagnostics for GGUF model compatibility.
  """

  @supported_architectures ["llama"]
  @supported_tensor_types %{
    0 => "F32",
    1 => "F16",
    2 => "Q4_0",
    3 => "Q4_1",
    6 => "Q5_0",
    7 => "Q5_1",
    8 => "Q8_0",
    9 => "Q8_1",
    10 => "Q2_K",
    11 => "Q3_K",
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
    chat_template = chat_template_status(gguf.metadata)
    missing_chat_template_tokens = missing_chat_template_tokens(gguf.metadata)

    %{
      version: gguf.version,
      tensor_count: gguf.tensor_count,
      metadata_count: gguf.metadata_count,
      architecture: metadata_value(gguf.metadata, "general.architecture"),
      supported_architectures: @supported_architectures,
      architecture_supported?: architecture_supported?(gguf.metadata),
      tokenizer_supported?: tokenizer_supported?(gguf.metadata),
      tokenizer_kind: tokenizer_kind(gguf.metadata),
      tokenizer_token_count: tokenizer_token_count(gguf.metadata),
      tokenizer_merge_count: tokenizer_merge_count(gguf.metadata),
      special_tokens: special_tokens(gguf.metadata),
      chat_template: chat_template,
      chat_usable: chat_usable?(chat_template, missing_chat_template_tokens),
      missing_chat_template_tokens: missing_chat_template_tokens,
      tensor_element_count: tensor_element_count(gguf.tensors),
      tensor_shapes: tensor_shapes(gguf.tensors),
      eager_f32_bytes: eager_f32_bytes(gguf.tensors),
      supported_tensor_types: supported_tensor_types(gguf.tensors),
      unsupported_tensor_types: unsupported_tensor_types(gguf.tensors),
      unsupported_tensors: unsupported_tensors(gguf.tensors),
      compatibility_issues: compatibility_issues(gguf.metadata, gguf.tensors),
      loadable?: loadable?(gguf.metadata, gguf.tensors)
    }
  end

  def format(%{} = diagnostic) do
    [
      "GGUF v#{diagnostic.version}",
      "architecture: #{diagnostic.architecture || "unknown"}",
      "architecture supported: #{diagnostic.architecture_supported?}",
      "tokenizer supported: #{diagnostic.tokenizer_supported?}",
      "loadable: #{diagnostic.loadable?}",
      "compatibility issues: #{format_compatibility_issues(diagnostic.compatibility_issues)}",
      "metadata: #{diagnostic.metadata_count}",
      "tensors: #{diagnostic.tensor_count}",
      "tokenizer kind: #{diagnostic.tokenizer_kind}",
      "tokenizer tokens: #{diagnostic.tokenizer_token_count || "unknown"}",
      "tokenizer merges: #{diagnostic.tokenizer_merge_count}",
      "special tokens: #{format_special_tokens(diagnostic.special_tokens)}",
      "chat template: #{diagnostic.chat_template}",
      "chat usable: #{diagnostic.chat_usable}",
      format_missing_chat_template_tokens(diagnostic.missing_chat_template_tokens),
      "tensor elements: #{diagnostic.tensor_element_count}",
      "tensor shapes: #{format_tensor_shapes(diagnostic.tensor_shapes)}",
      "eager f32 lower bound: #{format_bytes(diagnostic.eager_f32_bytes)}",
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

  defp architecture_supported?(metadata) do
    metadata
    |> metadata_value("general.architecture")
    |> then(&(&1 in @supported_architectures))
  end

  defp tokenizer_supported?(metadata) do
    match?(%{values: [_first | _rest]}, metadata_value(metadata, "tokenizer.ggml.tokens"))
  end

  defp tokenizer_kind(metadata) do
    case metadata_value(metadata, "tokenizer.ggml.merges") do
      %{values: [_first | _rest]} -> "bpe"
      _other -> "whitespace"
    end
  end

  defp tokenizer_merge_count(metadata) do
    case metadata_value(metadata, "tokenizer.ggml.merges") do
      %{values: values} -> length(values)
      _other -> 0
    end
  end

  defp loadable?(metadata, tensors) do
    architecture_supported?(metadata) and tokenizer_supported?(metadata) and
      unsupported_tensors(tensors) == []
  end

  defp compatibility_issues(metadata, tensors) do
    []
    |> add_architecture_issue(metadata)
    |> add_tokenizer_issue(metadata)
    |> add_tensor_type_issues(tensors)
    |> Enum.reverse()
  end

  defp add_architecture_issue(issues, metadata) do
    if architecture_supported?(metadata) do
      issues
    else
      architecture = metadata_value(metadata, "general.architecture") || "unknown"
      ["unsupported architecture: #{architecture}" | issues]
    end
  end

  defp add_tokenizer_issue(issues, metadata) do
    if tokenizer_supported?(metadata) do
      issues
    else
      ["missing tokenizer.ggml.tokens" | issues]
    end
  end

  defp add_tensor_type_issues(issues, tensors) do
    tensors
    |> unsupported_tensor_types()
    |> Enum.sort()
    |> Enum.reduce(issues, fn {type, count}, issues ->
      ["unsupported tensor type: #{type} (#{count})" | issues]
    end)
  end

  defp tensor_element_count(tensors) do
    tensors
    |> Enum.map(fn tensor -> Enum.product(tensor.dimensions) end)
    |> Enum.sum()
  end

  defp eager_f32_bytes(tensors), do: tensor_element_count(tensors) * 4

  defp tensor_shapes(tensors) do
    interesting =
      MapSet.new([
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight"
      ])

    tensors
    |> Enum.filter(&MapSet.member?(interesting, &1.name))
    |> Enum.map(fn tensor ->
      %{
        name: tensor.name,
        type: tensor_type_name(tensor.type),
        dimensions: tensor.dimensions,
        schema_shape: schema_shape(tensor.dimensions)
      }
    end)
  end

  defp type_counts(tensors) do
    tensors
    |> Enum.frequencies_by(& &1.type)
    |> Map.new(fn {type, count} -> {tensor_type_name(type), count} end)
  end

  defp tensor_type_name(type), do: Map.get(@supported_tensor_types, type, "type_#{type}")

  defp chat_template_status(metadata) do
    case metadata_value(metadata, "tokenizer.chat_template") do
      nil ->
        "none"

      template ->
        if Llamex.ChatTemplate.supported?(template), do: "supported", else: "unsupported"
    end
  end

  defp chat_usable?("supported", []), do: true
  defp chat_usable?(_chat_template, _missing_tokens), do: false

  defp missing_chat_template_tokens(metadata) do
    template = metadata_value(metadata, "tokenizer.chat_template")

    tokens =
      case metadata_value(metadata, "tokenizer.ggml.tokens") do
        %{values: values} -> MapSet.new(values)
        _other -> MapSet.new()
      end

    template
    |> Llamex.ChatTemplate.markers()
    |> Enum.reject(&MapSet.member?(tokens, &1))
  end

  defp tokenizer_token_count(metadata) do
    case metadata_value(metadata, "tokenizer.ggml.tokens") do
      %{values: values} -> length(values)
      _other -> nil
    end
  end

  defp special_tokens(metadata) do
    tokens =
      case metadata_value(metadata, "tokenizer.ggml.tokens") do
        %{values: values} -> values
        _other -> []
      end

    %{}
    |> put_special_token(metadata, tokens, :unknown, "tokenizer.ggml.unknown_token_id")
    |> put_special_token(metadata, tokens, :bos, "tokenizer.ggml.bos_token_id")
    |> put_special_token(metadata, tokens, :eos, "tokenizer.ggml.eos_token_id")
    |> put_special_token(metadata, tokens, :padding, "tokenizer.ggml.padding_token_id")
  end

  defp put_special_token(attrs, metadata, tokens, name, key) do
    case metadata_value(metadata, key) do
      id when is_integer(id) ->
        Map.put(attrs, name, %{id: id, piece: Enum.at(tokens, id)})

      _other ->
        attrs
    end
  end

  defp metadata_value(metadata, key) do
    case Map.fetch(metadata, key) do
      {:ok, %{value: value}} -> value
      :error -> nil
    end
  end

  defp format_missing_chat_template_tokens([]), do: "chat template missing tokens: none"

  defp format_missing_chat_template_tokens(tokens) do
    "chat template missing tokens: " <> Enum.join(tokens, ", ")
  end

  defp format_type_counts(counts) when map_size(counts) == 0, do: "none"

  defp format_type_counts(counts) do
    counts
    |> Enum.sort()
    |> Enum.map(fn {type, count} -> "#{type}=#{count}" end)
    |> Enum.join(", ")
  end

  defp format_compatibility_issues([]), do: "none"

  defp format_compatibility_issues(issues), do: Enum.join(issues, "; ")

  defp format_special_tokens(tokens) when map_size(tokens) == 0, do: "none"

  defp format_special_tokens(tokens) do
    tokens
    |> Enum.sort()
    |> Enum.map(fn {name, %{id: id, piece: piece}} -> "#{name}=#{id}:#{piece}" end)
    |> Enum.join(", ")
  end

  defp format_bytes(bytes) when bytes < 1024, do: "#{bytes} B"

  defp format_bytes(bytes) when bytes < 1024 * 1024 do
    "#{Float.round(bytes / 1024, 1)} KiB"
  end

  defp format_bytes(bytes) when bytes < 1024 * 1024 * 1024 do
    "#{Float.round(bytes / 1024 / 1024, 1)} MiB"
  end

  defp format_bytes(bytes), do: "#{Float.round(bytes / 1024 / 1024 / 1024, 1)} GiB"

  defp format_tensor_shapes([]), do: "none"

  defp format_tensor_shapes(tensors) do
    tensors
    |> Enum.map(fn tensor ->
      "#{tensor.name}=#{tensor.type} gguf:#{inspect(tensor.dimensions)} schema:#{inspect(tensor.schema_shape)}"
    end)
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

  defp schema_shape([columns, rows]), do: [rows, columns]
  defp schema_shape(dimensions), do: dimensions
end
