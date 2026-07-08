defmodule Llamex.GGUF.Diagnostic do
  @moduledoc """
  Diagnostics for GGUF model compatibility.
  """

  @supported_architectures ["llama"]
  @supported_tokenizers ["whitespace", "bpe"]
  @supported_tokenizer_models ["llama", "gpt2"]
  @supported_pre_tokenizers ["default", "gpt2", "llama-bpe"]
  @required_metadata_keys ["llama.embedding_length"]
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
    15 => "Q8_K",
    30 => "BF16"
  }

  def supported_architectures, do: @supported_architectures

  def supported_tokenizers, do: @supported_tokenizers

  def supported_tokenizer_models, do: @supported_tokenizer_models

  def supported_pre_tokenizers, do: @supported_pre_tokenizers

  def supported_tensor_type_names do
    @supported_tensor_types
    |> Map.values()
    |> Enum.sort()
  end

  def supported_tensor_type_ids do
    @supported_tensor_types
  end

  def supported_combinations do
    [
      %{
        architecture: "llama",
        tokenizers: supported_tokenizers(),
        tokenizer_models: supported_tokenizer_models(),
        pre_tokenizers: supported_pre_tokenizers(),
        tensor_types: supported_tensor_type_names()
      }
    ]
  end

  def supported_surface do
    %{
      supported_architectures: supported_architectures(),
      supported_tokenizers: supported_tokenizers(),
      supported_tokenizer_models: supported_tokenizer_models(),
      supported_pre_tokenizers: supported_pre_tokenizers(),
      supported_tensor_type_names: supported_tensor_type_names(),
      supported_tensor_type_ids: supported_tensor_type_ids(),
      supported_combinations: supported_combinations()
    }
  end

  def format_supported_surface(%{} = surface \\ supported_surface()) do
    [
      "supported architectures: #{Enum.join(surface.supported_architectures, ", ")}",
      "supported tokenizers: #{Enum.join(surface.supported_tokenizers, ", ")}",
      "supported tokenizer models: #{Enum.join(surface.supported_tokenizer_models, ", ")}",
      "supported pre-tokenizers: #{Enum.join(surface.supported_pre_tokenizers, ", ")}",
      "supported tensor type names: #{Enum.join(surface.supported_tensor_type_names, ", ")}",
      "supported combinations: #{format_supported_combinations(surface.supported_combinations)}"
    ]
    |> Enum.join("\n")
  end

  def inspect_file(path) when is_binary(path) do
    path
    |> File.read!()
    |> inspect_binary()
  end

  def inspect_binary(binary) when is_binary(binary) do
    gguf = Llamex.GGUF.Reader.read_binary(binary)
    inspect_reader(gguf)
  end

  def inspect_reader(%Llamex.GGUF.Reader{} = gguf) do
    chat_template = chat_template_status(gguf.metadata)
    missing_chat_template_tokens = missing_chat_template_tokens(gguf.metadata)

    %{
      version: gguf.version,
      tensor_count: gguf.tensor_count,
      metadata_count: gguf.metadata_count,
      architecture: metadata_value(gguf.metadata, "general.architecture"),
      supported_architectures: supported_architectures(),
      supported_combinations: supported_combinations(),
      architecture_supported?: architecture_supported?(gguf.metadata),
      tokenizer_supported?: tokenizer_supported?(gguf.metadata),
      tokenizer_model: tokenizer_model(gguf.metadata),
      tokenizer_model_supported?: tokenizer_model_supported?(gguf.metadata),
      pre_tokenizer: pre_tokenizer(gguf.metadata),
      pre_tokenizer_supported?: pre_tokenizer_supported?(gguf.metadata),
      missing_required_metadata: missing_required_metadata(gguf.metadata),
      tokenizer_kind: tokenizer_kind(gguf.metadata),
      supported_tokenizers: supported_tokenizers(),
      supported_tokenizer_models: supported_tokenizer_models(),
      supported_pre_tokenizers: supported_pre_tokenizers(),
      tokenizer_token_count: tokenizer_token_count(gguf.metadata),
      tokenizer_merge_count: tokenizer_merge_count(gguf.metadata),
      special_tokens: special_tokens(gguf.metadata),
      chat_template: chat_template,
      chat_usable: chat_usable?(chat_template, missing_chat_template_tokens),
      missing_chat_template_tokens: missing_chat_template_tokens,
      tensor_element_count: tensor_element_count(gguf.tensors),
      tensor_shapes: tensor_shapes(gguf.tensors),
      eager_f32_bytes: eager_f32_bytes(gguf.tensors),
      supported_tensor_type_names: supported_tensor_type_names(),
      supported_tensor_type_ids: supported_tensor_type_ids(),
      supported_tensor_types: supported_tensor_types(gguf.tensors),
      unsupported_tensor_types: unsupported_tensor_types(gguf.tensors),
      unsupported_tensors: unsupported_tensors(gguf.tensors),
      compatibility_issues: compatibility_issues(gguf.metadata, gguf.tensors),
      loadable?: loadable?(gguf.metadata, gguf.tensors)
    }
  end

  def loadable?(%Llamex.GGUF.Reader{} = gguf) do
    loadable?(gguf.metadata, gguf.tensors)
  end

  def compatibility_issues(%Llamex.GGUF.Reader{} = gguf) do
    compatibility_issues(gguf.metadata, gguf.tensors)
  end

  def format(%{} = diagnostic) do
    [
      "GGUF v#{diagnostic.version}",
      "architecture: #{diagnostic.architecture || "unknown"}",
      "supported architectures: #{Enum.join(diagnostic.supported_architectures, ", ")}",
      "supported combinations: #{format_supported_combinations(diagnostic.supported_combinations)}",
      "architecture supported: #{diagnostic.architecture_supported?}",
      "supported tokenizers: #{Enum.join(diagnostic.supported_tokenizers, ", ")}",
      "tokenizer supported: #{diagnostic.tokenizer_supported?}",
      "supported tokenizer models: #{Enum.join(diagnostic.supported_tokenizer_models, ", ")}",
      "tokenizer model supported: #{diagnostic.tokenizer_model_supported?}",
      "supported pre-tokenizers: #{Enum.join(diagnostic.supported_pre_tokenizers, ", ")}",
      "pre-tokenizer supported: #{diagnostic.pre_tokenizer_supported?}",
      "missing required metadata: #{format_missing_required_metadata(diagnostic.missing_required_metadata)}",
      "loadable: #{diagnostic.loadable?}",
      "compatibility issues: #{format_compatibility_issues(diagnostic.compatibility_issues)}",
      "metadata: #{diagnostic.metadata_count}",
      "tensors: #{diagnostic.tensor_count}",
      "tokenizer model: #{diagnostic.tokenizer_model || "unknown"}",
      "pre-tokenizer: #{diagnostic.pre_tokenizer || "unknown"}",
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
      "supported tensor type names: #{Enum.join(diagnostic.supported_tensor_type_names, ", ")}",
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

  defp tokenizer_model(metadata), do: metadata_value(metadata, "tokenizer.ggml.model")

  defp tokenizer_model_supported?(metadata) do
    case tokenizer_model(metadata) do
      nil -> true
      model -> model in @supported_tokenizer_models
    end
  end

  defp pre_tokenizer(metadata), do: metadata_value(metadata, "tokenizer.ggml.pre")

  defp pre_tokenizer_supported?(metadata) do
    case pre_tokenizer(metadata) do
      nil -> true
      pre_tokenizer -> pre_tokenizer in @supported_pre_tokenizers
    end
  end

  defp missing_required_metadata(metadata) do
    Enum.reject(@required_metadata_keys, &Map.has_key?(metadata, &1))
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
      tokenizer_model_supported?(metadata) and pre_tokenizer_supported?(metadata) and
      missing_required_metadata(metadata) == [] and
      unsupported_tensors(tensors) == []
  end

  defp compatibility_issues(metadata, tensors) do
    []
    |> add_architecture_issue(metadata)
    |> add_tokenizer_issue(metadata)
    |> add_tokenizer_model_issue(metadata)
    |> add_pre_tokenizer_issue(metadata)
    |> add_required_metadata_issues(metadata)
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

  defp add_tokenizer_model_issue(issues, metadata) do
    case tokenizer_model(metadata) do
      nil ->
        issues

      model ->
        if tokenizer_model_supported?(metadata) do
          issues
        else
          ["unsupported tokenizer model: #{model}" | issues]
        end
    end
  end

  defp add_pre_tokenizer_issue(issues, metadata) do
    case pre_tokenizer(metadata) do
      nil ->
        issues

      pre_tokenizer ->
        if pre_tokenizer_supported?(metadata) do
          issues
        else
          ["unsupported pre-tokenizer: #{pre_tokenizer}" | issues]
        end
    end
  end

  defp add_required_metadata_issues(issues, metadata) do
    metadata
    |> missing_required_metadata()
    |> Enum.reduce(issues, fn key, issues ->
      ["missing required metadata: #{key}" | issues]
    end)
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

  defp format_supported_combinations(combinations) do
    combinations
    |> Enum.map(fn combination ->
      tokenizers = Enum.join(combination.tokenizers, "/")
      tokenizer_models = Enum.join(combination.tokenizer_models, "/")
      pre_tokenizers = Enum.join(combination.pre_tokenizers, "/")
      tensor_types = Enum.join(combination.tensor_types, "/")

      "#{combination.architecture}+#{tokenizers}+#{tokenizer_models}+#{pre_tokenizers}+#{tensor_types}"
    end)
    |> Enum.join("; ")
  end

  defp format_compatibility_issues([]), do: "none"

  defp format_compatibility_issues(issues), do: Enum.join(issues, "; ")

  defp format_missing_required_metadata([]), do: "none"

  defp format_missing_required_metadata(keys), do: Enum.join(keys, ", ")

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
