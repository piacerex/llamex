defmodule Llamex.GGUF.Diagnostic do
  @moduledoc """
  Diagnostics for GGUF model compatibility.
  """

  @known_architectures ["llama", "gemma3"]
  @supported_architectures ["llama"]
  @supported_tokenizers ["whitespace", "bpe"]
  @supported_tokenizer_models ["llama", "gpt2"]
  @supported_pre_tokenizers ["default", "gpt2", "llama-bpe"]
  @unsupported_feature_metadata [
    "*.attention.sliding_window",
    "*.rope.scaling.type"
  ]
  @unsupported_feature_detail_suffixes [
    "attention.sliding_window",
    "rope.scaling.type",
    "rope.scaling.factor",
    "rope.scaling.original_context_length"
  ]
  @required_metadata_suffixes ["embedding_length"]
  @summary_keys [
    :architecture,
    :architecture_runtime_status,
    :loadable?,
    :compatibility_issues,
    :compatibility_issue_groups,
    :blocking_issue_groups,
    :tokenizer_model,
    :tokenizer_model_supported?,
    :pre_tokenizer,
    :pre_tokenizer_supported?,
    :chat_usable,
    :chat_template_family,
    :missing_chat_template_tokens,
    :chat_template_issues,
    :tokenizer_metadata_issues,
    :model_config_metadata_prefix,
    :missing_model_config_metadata,
    :unsupported_tensor_features,
    :tensor_schema_mappings,
    :tensor_schema_issues,
    :eager_f32_bytes,
    :gguf_payload_bytes,
    :eager_f32_expansion_ratio,
    :tensor_payload_by_type,
    :top_tensor_payloads
  ]
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

  def known_architectures, do: @known_architectures

  def supported_tokenizers, do: @supported_tokenizers

  def supported_tokenizer_models, do: @supported_tokenizer_models

  def supported_pre_tokenizers, do: @supported_pre_tokenizers

  def supported_chat_templates, do: Llamex.ChatTemplate.supported_families()

  def unsupported_feature_metadata, do: @unsupported_feature_metadata

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
        runtime_status: "supported",
        tokenizers: supported_tokenizers(),
        tokenizer_models: supported_tokenizer_models(),
        pre_tokenizers: supported_pre_tokenizers(),
        tensor_types: supported_tensor_type_names()
      }
    ]
  end

  def known_combinations do
    runtime_surface = architecture_runtime_surface()

    Enum.map(known_architectures(), fn architecture ->
      %{
        architecture: architecture,
        runtime_status: Map.fetch!(runtime_surface, architecture),
        tokenizers: supported_tokenizers(),
        tokenizer_models: supported_tokenizer_models(),
        pre_tokenizers: supported_pre_tokenizers(),
        tensor_types: supported_tensor_type_names()
      }
    end)
  end

  def architecture_runtime_surface do
    Map.new(known_architectures(), fn architecture ->
      status =
        if architecture in supported_architectures() do
          "supported"
        else
          "known_unsupported"
        end

      {architecture, status}
    end)
  end

  def tokenizer_metadata_surface do
    Map.new(known_architectures(), fn architecture ->
      {
        architecture,
        %{
          tokenizer_models: supported_tokenizer_models(),
          pre_tokenizers: supported_pre_tokenizers()
        }
      }
    end)
  end

  def supported_surface do
    %{
      supported_architectures: supported_architectures(),
      known_architectures: known_architectures(),
      architecture_runtime_surface: architecture_runtime_surface(),
      supported_tokenizers: supported_tokenizers(),
      supported_tokenizer_models: supported_tokenizer_models(),
      supported_pre_tokenizers: supported_pre_tokenizers(),
      tokenizer_metadata_surface: tokenizer_metadata_surface(),
      supported_chat_templates: supported_chat_templates(),
      unsupported_feature_metadata: unsupported_feature_metadata(),
      model_config_surface: Llamex.GGUF.ModelConfig.surface(known_architectures()),
      tensor_schema_surface: Llamex.GGUF.TensorSchema.surface(known_architectures()),
      supported_tensor_type_names: supported_tensor_type_names(),
      supported_tensor_type_ids: supported_tensor_type_ids(),
      known_combinations: known_combinations(),
      supported_combinations: supported_combinations()
    }
  end

  def format_supported_surface(%{} = surface \\ supported_surface()) do
    [
      "supported architectures: #{Enum.join(surface.supported_architectures, ", ")}",
      "known architectures: #{Enum.join(surface.known_architectures, ", ")}",
      "architecture runtime surface: #{format_architecture_runtime_surface(surface.architecture_runtime_surface)}",
      "supported tokenizers: #{Enum.join(surface.supported_tokenizers, ", ")}",
      "supported tokenizer models: #{Enum.join(surface.supported_tokenizer_models, ", ")}",
      "supported pre-tokenizers: #{Enum.join(surface.supported_pre_tokenizers, ", ")}",
      "tokenizer metadata surface: #{format_tokenizer_metadata_surface(surface.tokenizer_metadata_surface)}",
      "supported chat templates: #{Enum.join(surface.supported_chat_templates, ", ")}",
      "unsupported feature metadata: #{Enum.join(surface.unsupported_feature_metadata, ", ")}",
      "model config surface: #{format_model_config_surface(surface.model_config_surface)}",
      "tensor schema surface: #{format_tensor_schema_surface(surface.tensor_schema_surface)}",
      "supported tensor type names: #{Enum.join(surface.supported_tensor_type_names, ", ")}",
      "supported tensor type ids: #{format_supported_tensor_type_ids(surface.supported_tensor_type_ids)}",
      "known combinations: #{format_supported_combinations(surface.known_combinations, runtime_status?: true)}",
      "supported combinations: #{format_supported_combinations(surface.supported_combinations)}"
    ]
    |> Enum.join("\n")
  end

  def inspect_file(path) when is_binary(path) do
    path
    |> File.read!()
    |> inspect_binary()
  end

  def inspect_summary_file(path) when is_binary(path) do
    path
    |> inspect_file()
    |> summary()
  end

  def inspect_binary(binary) when is_binary(binary) do
    gguf = Llamex.GGUF.Reader.read_binary(binary)
    inspect_reader(gguf)
  end

  def inspect_reader(%Llamex.GGUF.Reader{} = gguf) do
    chat_template = chat_template_status(gguf.metadata)
    chat_template_family = chat_template_family(gguf.metadata)
    missing_chat_template_tokens = missing_chat_template_tokens(gguf.metadata)

    %{
      version: gguf.version,
      tensor_count: gguf.tensor_count,
      metadata_count: gguf.metadata_count,
      architecture: metadata_value(gguf.metadata, "general.architecture"),
      known_architectures: known_architectures(),
      supported_architectures: supported_architectures(),
      supported_combinations: supported_combinations(),
      architecture_known?: architecture_known?(gguf.metadata),
      architecture_supported?: architecture_supported?(gguf.metadata),
      architecture_runtime_status: architecture_runtime_status(gguf.metadata),
      tokenizer_supported?: tokenizer_supported?(gguf.metadata),
      tokenizer_metadata: tokenizer_metadata_for(gguf.metadata),
      tokenizer_model: tokenizer_model(gguf.metadata),
      tokenizer_model_supported?: tokenizer_model_supported?(gguf.metadata),
      pre_tokenizer: pre_tokenizer(gguf.metadata),
      pre_tokenizer_supported?: pre_tokenizer_supported?(gguf.metadata),
      missing_required_metadata: missing_required_metadata(gguf.metadata),
      model_config_metadata_prefix: metadata_prefix(gguf.metadata),
      model_config: model_config(gguf.metadata),
      missing_model_config_metadata: missing_model_config_metadata(gguf.metadata),
      tokenizer_kind: tokenizer_kind(gguf.metadata),
      supported_tokenizers: supported_tokenizers(),
      supported_tokenizer_models: supported_tokenizer_models(),
      supported_pre_tokenizers: supported_pre_tokenizers(),
      tokenizer_token_count: tokenizer_token_count(gguf.metadata),
      tokenizer_merge_count: tokenizer_merge_count(gguf.metadata),
      tokenizer_score_count: tokenizer_score_count(gguf.metadata),
      tokenizer_metadata_issues: tokenizer_metadata_issues(gguf.metadata),
      tokenizer_token_types: tokenizer_token_types(gguf.metadata),
      special_tokens: special_tokens(gguf.metadata),
      unsupported_features: unsupported_features(gguf.metadata),
      unsupported_feature_metadata_values: unsupported_feature_metadata_values(gguf.metadata),
      unsupported_tensor_features: unsupported_tensor_features(gguf.metadata, gguf.tensors),
      chat_template: chat_template,
      chat_template_family: chat_template_family,
      chat_usable: chat_usable?(chat_template, missing_chat_template_tokens),
      missing_chat_template_tokens: missing_chat_template_tokens,
      chat_template_issues: chat_template_issues(chat_template, missing_chat_template_tokens),
      tensor_element_count: tensor_element_count(gguf.tensors),
      tensor_schema_mappings: tensor_schema_mappings(gguf.metadata, gguf.tensors),
      tensor_schema_issues: tensor_schema_issues(gguf.metadata, gguf.tensors),
      tensor_shapes: tensor_shapes(gguf.metadata, gguf.tensors),
      eager_f32_bytes: eager_f32_bytes(gguf.tensors),
      gguf_payload_bytes: gguf_payload_bytes(gguf.tensors),
      eager_f32_expansion_ratio: eager_f32_expansion_ratio(gguf.tensors),
      tensor_payload_by_type: tensor_payload_by_type(gguf.tensors),
      top_tensor_payloads: top_tensor_payloads(gguf.tensors),
      missing_required_tensors: missing_required_tensors(gguf.metadata, gguf.tensors),
      tensor_shape_issues: tensor_shape_issues(gguf.metadata, gguf.tensors),
      supported_tensor_type_names: supported_tensor_type_names(),
      supported_tensor_type_ids: supported_tensor_type_ids(),
      supported_tensor_types: supported_tensor_types(gguf.tensors),
      supported_tensors: supported_tensors(gguf.tensors),
      unsupported_tensor_types: unsupported_tensor_types(gguf.tensors),
      unsupported_tensors: unsupported_tensors(gguf.tensors),
      compatibility_issues: compatibility_issues(gguf.metadata, gguf.tensors),
      compatibility_issue_groups: compatibility_issue_groups(gguf.metadata, gguf.tensors),
      blocking_issue_groups:
        gguf.metadata
        |> compatibility_issue_groups(gguf.tensors)
        |> blocking_issue_groups(),
      loadable?: loadable?(gguf.metadata, gguf.tensors)
    }
  end

  def loadable?(%Llamex.GGUF.Reader{} = gguf) do
    loadable?(gguf.metadata, gguf.tensors)
  end

  def compatibility_issues(%Llamex.GGUF.Reader{} = gguf) do
    compatibility_issues(gguf.metadata, gguf.tensors)
  end

  def summary(%{} = diagnostic), do: Map.take(diagnostic, @summary_keys)

  def format(%{} = diagnostic) do
    [
      "GGUF v#{diagnostic.version}",
      "architecture: #{diagnostic.architecture || "unknown"}",
      "supported architectures: #{Enum.join(diagnostic.supported_architectures, ", ")}",
      "known architectures: #{Enum.join(diagnostic.known_architectures, ", ")}",
      "supported combinations: #{format_supported_combinations(diagnostic.supported_combinations)}",
      "architecture known: #{diagnostic.architecture_known?}",
      "architecture supported: #{diagnostic.architecture_supported?}",
      "architecture runtime status: #{diagnostic.architecture_runtime_status}",
      "supported tokenizers: #{Enum.join(diagnostic.supported_tokenizers, ", ")}",
      "tokenizer supported: #{diagnostic.tokenizer_supported?}",
      "supported tokenizer models: #{Enum.join(diagnostic.supported_tokenizer_models, ", ")}",
      "tokenizer model supported: #{diagnostic.tokenizer_model_supported?}",
      "supported pre-tokenizers: #{Enum.join(diagnostic.supported_pre_tokenizers, ", ")}",
      "pre-tokenizer supported: #{diagnostic.pre_tokenizer_supported?}",
      "missing required metadata: #{format_missing_required_metadata(diagnostic.missing_required_metadata)}",
      "model config metadata prefix: #{diagnostic.model_config_metadata_prefix}",
      "model config: #{format_model_config(diagnostic.model_config)}",
      "missing model config metadata: #{format_missing_model_config_metadata(diagnostic.missing_model_config_metadata)}",
      "loadable: #{diagnostic.loadable?}",
      "compatibility issues: #{format_compatibility_issues(diagnostic.compatibility_issues)}",
      "compatibility issue groups: #{format_compatibility_issue_groups(diagnostic.compatibility_issue_groups)}",
      "blocking issue groups: #{format_blocking_issue_groups(diagnostic.blocking_issue_groups)}",
      "metadata: #{diagnostic.metadata_count}",
      "tensors: #{diagnostic.tensor_count}",
      "tokenizer model: #{diagnostic.tokenizer_model || "unknown"}",
      "pre-tokenizer: #{diagnostic.pre_tokenizer || "unknown"}",
      "tokenizer metadata surface: #{format_tokenizer_metadata(diagnostic.tokenizer_metadata)}",
      "tokenizer kind: #{diagnostic.tokenizer_kind}",
      "tokenizer tokens: #{diagnostic.tokenizer_token_count || "unknown"}",
      "tokenizer merges: #{diagnostic.tokenizer_merge_count}",
      "tokenizer scores: #{diagnostic.tokenizer_score_count}",
      "tokenizer metadata issues: #{format_tokenizer_metadata_issues(diagnostic.tokenizer_metadata_issues)}",
      "tokenizer token types: #{format_type_counts(diagnostic.tokenizer_token_types)}",
      "special tokens: #{format_special_tokens(diagnostic.special_tokens)}",
      "unsupported features: #{format_unsupported_features(diagnostic.unsupported_features)}",
      "unsupported feature metadata values: #{format_metadata_values(diagnostic.unsupported_feature_metadata_values)}",
      "unsupported tensor features: #{format_unsupported_features(diagnostic.unsupported_tensor_features)}",
      "chat template: #{diagnostic.chat_template}",
      "chat template family: #{diagnostic.chat_template_family}",
      "chat usable: #{diagnostic.chat_usable}",
      format_missing_chat_template_tokens(diagnostic.missing_chat_template_tokens),
      "chat template issues: #{format_chat_template_issues(diagnostic.chat_template_issues)}",
      "tensor elements: #{diagnostic.tensor_element_count}",
      "tensor schema mappings: #{format_tensor_schema_mappings(diagnostic.tensor_schema_mappings)}",
      "tensor schema issues: #{format_tensor_schema_issues(diagnostic.tensor_schema_issues)}",
      "tensor shapes: #{format_tensor_shapes(diagnostic.tensor_shapes)}",
      "eager f32 lower bound: #{format_bytes(diagnostic.eager_f32_bytes)}",
      "gguf payload bytes: #{format_bytes(diagnostic.gguf_payload_bytes)}",
      "eager f32 expansion ratio: #{format_ratio(diagnostic.eager_f32_expansion_ratio)}",
      "tensor payload by type: #{format_tensor_payload_by_type(diagnostic.tensor_payload_by_type)}",
      format_top_tensor_payloads(diagnostic.top_tensor_payloads),
      "missing required tensors: #{format_missing_required_tensors(diagnostic.missing_required_tensors)}",
      "tensor shape issues: #{format_tensor_shape_issues(diagnostic.tensor_shape_issues)}",
      "supported tensor type names: #{Enum.join(diagnostic.supported_tensor_type_names, ", ")}",
      "supported tensor types: #{format_type_counts(diagnostic.supported_tensor_types)}",
      format_supported_tensors(diagnostic.supported_tensors),
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

  defp supported_tensors(tensors) do
    tensors
    |> Enum.filter(&supported_tensor_type?/1)
    |> Enum.map(fn tensor ->
      %{
        name: tensor.name,
        type: tensor.type,
        type_name: tensor_type_name(tensor.type),
        dimensions: tensor.dimensions
      }
    end)
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

  defp architecture_known?(metadata) do
    metadata
    |> metadata_value("general.architecture")
    |> then(&(&1 in @known_architectures))
  end

  defp architecture_supported?(metadata) do
    metadata
    |> metadata_value("general.architecture")
    |> then(&(&1 in @supported_architectures))
  end

  defp architecture_runtime_status(metadata) do
    cond do
      architecture_supported?(metadata) -> "supported"
      architecture_known?(metadata) -> "known_unsupported"
      true -> "unknown"
    end
  end

  defp tokenizer_supported?(metadata) do
    match?(%{values: [_first | _rest]}, metadata_value(metadata, "tokenizer.ggml.tokens"))
  end

  defp tokenizer_model(metadata), do: metadata_value(metadata, "tokenizer.ggml.model")

  defp tokenizer_model_supported?(metadata) do
    case tokenizer_model(metadata) do
      nil -> true
      model -> model in tokenizer_models_for(metadata)
    end
  end

  defp pre_tokenizer(metadata), do: metadata_value(metadata, "tokenizer.ggml.pre")

  defp pre_tokenizer_supported?(metadata) do
    case pre_tokenizer(metadata) do
      nil -> true
      pre_tokenizer -> pre_tokenizer in pre_tokenizers_for(metadata)
    end
  end

  defp tokenizer_models_for(metadata) do
    metadata
    |> tokenizer_metadata_for()
    |> Map.fetch!(:tokenizer_models)
  end

  defp pre_tokenizers_for(metadata) do
    metadata
    |> tokenizer_metadata_for()
    |> Map.fetch!(:pre_tokenizers)
  end

  defp tokenizer_metadata_for(metadata) do
    architecture = metadata_value(metadata, "general.architecture")

    tokenizer_metadata_surface()
    |> Map.get(architecture, %{
      tokenizer_models: supported_tokenizer_models(),
      pre_tokenizers: supported_pre_tokenizers()
    })
  end

  defp missing_required_metadata(metadata) do
    prefix = metadata_prefix(metadata)

    @required_metadata_suffixes
    |> Enum.map(&metadata_key(prefix, &1))
    |> Enum.reject(&Map.has_key?(metadata, &1))
  end

  defp model_config(metadata) do
    metadata
    |> Llamex.GGUF.ModelConfig.partial_from_metadata()
    |> Enum.map(fn {key, value} -> {String.to_atom(key), value} end)
    |> Map.new()
  end

  defp missing_model_config_metadata(metadata) do
    Llamex.GGUF.ModelConfig.missing_metadata(metadata)
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

  defp tokenizer_score_count(metadata) do
    case metadata_value(metadata, "tokenizer.ggml.scores") do
      %{values: values} -> length(values)
      _other -> 0
    end
  end

  defp tokenizer_metadata_issues(metadata) do
    []
    |> add_tokenizer_scores_issue(metadata)
    |> add_tokenizer_token_type_issue(metadata)
    |> add_chat_marker_token_type_issues(metadata)
    |> Enum.reverse()
  end

  defp add_tokenizer_scores_issue(issues, metadata) do
    case {metadata_value(metadata, "tokenizer.ggml.tokens"),
          metadata_value(metadata, "tokenizer.ggml.scores")} do
      {%{values: tokens}, %{values: scores}} when length(tokens) != length(scores) ->
        [
          "tokenizer score count mismatch: tokens=#{length(tokens)} scores=#{length(scores)}"
          | issues
        ]

      _other ->
        issues
    end
  end

  defp add_tokenizer_token_type_issue(issues, metadata) do
    case {metadata_value(metadata, "tokenizer.ggml.tokens"),
          metadata_value(metadata, "tokenizer.ggml.token_type")} do
      {%{values: tokens}, %{values: token_types}} when length(tokens) != length(token_types) ->
        [
          "tokenizer token_type count mismatch: tokens=#{length(tokens)} token_types=#{length(token_types)}"
          | issues
        ]

      _other ->
        issues
    end
  end

  defp add_chat_marker_token_type_issues(issues, metadata) do
    template = metadata_value(metadata, "tokenizer.chat_template")

    case {template, metadata_value(metadata, "tokenizer.ggml.tokens"),
          metadata_value(metadata, "tokenizer.ggml.token_type")} do
      {template, %{values: tokens}, %{values: token_types}} when is_binary(template) ->
        template
        |> Llamex.ChatTemplate.markers()
        |> Enum.reduce(issues, fn marker, issues ->
          case Enum.find_index(tokens, &(&1 == marker)) do
            nil ->
              issues

            index ->
              if Enum.at(token_types, index) == 3 do
                issues
              else
                ["chat marker token should be control: #{marker}" | issues]
              end
          end
        end)

      _other ->
        issues
    end
  end

  defp loadable?(metadata, tensors) do
    architecture_supported?(metadata) and tokenizer_supported?(metadata) and
      tokenizer_model_supported?(metadata) and pre_tokenizer_supported?(metadata) and
      missing_required_metadata(metadata) == [] and
      unsupported_features(metadata) == [] and
      unsupported_tensor_features(metadata, tensors) == [] and
      missing_required_tensors(metadata, tensors) == [] and
      tensor_schema_issues(metadata, tensors) == [] and
      tensor_shape_issues(metadata, tensors) == [] and
      unsupported_tensors(tensors) == []
  end

  defp compatibility_issues(metadata, tensors) do
    []
    |> add_architecture_issue(metadata)
    |> add_tokenizer_issue(metadata)
    |> add_tokenizer_model_issue(metadata)
    |> add_pre_tokenizer_issue(metadata)
    |> add_required_metadata_issues(metadata)
    |> add_unsupported_feature_issues(metadata)
    |> add_unsupported_tensor_feature_issues(metadata, tensors)
    |> add_required_tensor_issues(metadata, tensors)
    |> add_tensor_schema_issues(metadata, tensors)
    |> add_tensor_shape_issues(metadata, tensors)
    |> add_tensor_type_issues(tensors)
    |> Enum.reverse()
  end

  defp compatibility_issue_groups(metadata, tensors) do
    %{
      runtime: [] |> add_architecture_issue(metadata) |> Enum.reverse(),
      tokenizer:
        []
        |> add_tokenizer_issue(metadata)
        |> add_tokenizer_model_issue(metadata)
        |> add_pre_tokenizer_issue(metadata)
        |> Enum.reverse(),
      metadata: [] |> add_required_metadata_issues(metadata) |> Enum.reverse(),
      features: [] |> add_unsupported_feature_issues(metadata) |> Enum.reverse(),
      tensor_features:
        [] |> add_unsupported_tensor_feature_issues(metadata, tensors) |> Enum.reverse(),
      tensors:
        []
        |> add_required_tensor_issues(metadata, tensors)
        |> add_tensor_schema_issues(metadata, tensors)
        |> add_tensor_shape_issues(metadata, tensors)
        |> add_tensor_type_issues(tensors)
        |> Enum.reverse()
    }
  end

  defp blocking_issue_groups(groups) do
    [:runtime, :tokenizer, :metadata, :features, :tensor_features, :tensors]
    |> Enum.filter(fn group -> Map.fetch!(groups, group) != [] end)
  end

  defp add_architecture_issue(issues, metadata) do
    if architecture_supported?(metadata) do
      issues
    else
      architecture = metadata_value(metadata, "general.architecture") || "unknown"

      if architecture_known?(metadata) do
        ["unsupported architecture runtime: #{architecture}" | issues]
      else
        ["unsupported architecture: #{architecture}" | issues]
      end
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

  defp add_unsupported_feature_issues(issues, metadata) do
    metadata
    |> unsupported_features()
    |> Enum.reduce(issues, fn issue, issues -> [issue | issues] end)
  end

  defp add_unsupported_tensor_feature_issues(issues, metadata, tensors) do
    metadata
    |> unsupported_tensor_features(tensors)
    |> Enum.reduce(issues, fn issue, issues -> [issue | issues] end)
  end

  defp add_required_tensor_issues(issues, metadata, tensors) do
    tensors
    |> then(&missing_required_tensors(metadata, &1))
    |> Enum.reduce(issues, fn name, issues ->
      ["missing required tensor: #{name}" | issues]
    end)
  end

  defp add_tensor_shape_issues(issues, metadata, tensors) do
    metadata
    |> tensor_shape_issues(tensors)
    |> Enum.reduce(issues, fn issue, issues ->
      [issue | issues]
    end)
  end

  defp add_tensor_schema_issues(issues, metadata, tensors) do
    metadata
    |> tensor_schema_issues(tensors)
    |> Enum.reduce(issues, fn issue, issues ->
      [issue | issues]
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

  defp tensor_schema_mappings(metadata, tensors) do
    architecture = metadata_value(metadata, "general.architecture")

    Llamex.GGUF.TensorSchema.mappings(architecture, Enum.map(tensors, & &1.name))
  end

  defp tensor_schema_issues(metadata, tensors) do
    architecture = metadata_value(metadata, "general.architecture")

    Llamex.GGUF.TensorSchema.unmapped_schema_issues(architecture, Enum.map(tensors, & &1.name))
  end

  defp unsupported_tensor_features(metadata, tensors) do
    architecture = metadata_value(metadata, "general.architecture")

    Llamex.GGUF.TensorSchema.unsupported_feature_issues(
      architecture,
      Enum.map(tensors, & &1.name)
    )
  end

  defp missing_required_tensors(metadata, tensors) do
    tensor_names = MapSet.new(Enum.map(tensors, & &1.name))
    architecture = metadata_value(metadata, "general.architecture")

    architecture
    |> Llamex.GGUF.TensorSchema.required_tensor_names()
    |> Enum.reject(&MapSet.member?(tensor_names, &1))
  end

  defp tensor_shape_issues(metadata, tensors) do
    architecture = metadata_value(metadata, "general.architecture")

    case {metadata_value(metadata, metadata_key(metadata_prefix(metadata), "embedding_length")),
          find_tensor(tensors, Llamex.GGUF.TensorSchema.token_embedding_name(architecture))} do
      {embedding_length, %{dimensions: dimensions}} when is_integer(embedding_length) ->
        case Llamex.GGUF.TensorSchema.schema_shape(dimensions) do
          [_vocab_size, ^embedding_length] ->
            []

          shape ->
            [
              "tensor shape mismatch: token_embd.weight schema #{inspect(shape)} expected embedding length #{embedding_length}"
            ]
        end

      _other ->
        []
    end
  end

  defp find_tensor(tensors, name), do: Enum.find(tensors, &(&1.name == name))

  defp eager_f32_bytes(tensors), do: tensor_element_count(tensors) * 4

  defp gguf_payload_bytes(tensors) do
    tensors
    |> Enum.map(&tensor_payload_bytes/1)
    |> Enum.sum()
  end

  defp eager_f32_expansion_ratio(tensors) do
    payload_bytes = gguf_payload_bytes(tensors)

    if payload_bytes == 0 do
      nil
    else
      eager_f32_bytes(tensors) / payload_bytes
    end
  end

  defp tensor_payload_by_type(tensors) do
    tensors
    |> Enum.group_by(&tensor_type_name(&1.type))
    |> Map.new(fn {type_name, tensors} ->
      eager_f32_bytes = eager_f32_bytes(tensors)
      gguf_payload_bytes = gguf_payload_bytes(tensors)

      {type_name,
       %{
         tensors: length(tensors),
         elements: tensor_element_count(tensors),
         eager_f32_bytes: eager_f32_bytes,
         gguf_payload_bytes: gguf_payload_bytes,
         eager_f32_expansion_ratio: expansion_ratio(eager_f32_bytes, gguf_payload_bytes)
       }}
    end)
  end

  defp expansion_ratio(_eager_f32_bytes, 0), do: nil

  defp expansion_ratio(eager_f32_bytes, gguf_payload_bytes),
    do: eager_f32_bytes / gguf_payload_bytes

  defp top_tensor_payloads(tensors) do
    tensors
    |> Enum.map(&tensor_payload_summary/1)
    |> Enum.sort_by(fn tensor ->
      {-tensor.eager_f32_bytes, -tensor.gguf_payload_bytes, tensor.name}
    end)
    |> Enum.take(10)
  end

  defp tensor_payload_summary(tensor) do
    elements = element_count(tensor.dimensions)
    eager_f32_bytes = elements * 4
    gguf_payload_bytes = tensor_payload_bytes(tensor)

    %{
      name: tensor.name,
      type: tensor_type_name(tensor.type),
      dimensions: tensor.dimensions,
      elements: elements,
      eager_f32_bytes: eager_f32_bytes,
      gguf_payload_bytes: gguf_payload_bytes,
      eager_f32_expansion_ratio: expansion_ratio(eager_f32_bytes, gguf_payload_bytes)
    }
  end

  defp tensor_shapes(metadata, tensors) do
    architecture = metadata_value(metadata, "general.architecture")

    interesting =
      architecture
      |> Llamex.GGUF.TensorSchema.interesting_tensor_names()
      |> MapSet.new()

    tensors
    |> Enum.filter(&MapSet.member?(interesting, &1.name))
    |> Enum.map(fn tensor ->
      %{
        name: tensor.name,
        type: tensor_type_name(tensor.type),
        dimensions: tensor.dimensions,
        schema_shape: Llamex.GGUF.TensorSchema.schema_shape(tensor.dimensions)
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

  defp chat_template_family(metadata) do
    metadata
    |> metadata_value("tokenizer.chat_template")
    |> Llamex.ChatTemplate.family()
  end

  defp chat_usable?("supported", []), do: true
  defp chat_usable?(_chat_template, _missing_tokens), do: false

  defp chat_template_issues("none", _missing_tokens), do: []

  defp chat_template_issues("unsupported", _missing_tokens), do: ["unsupported chat template"]

  defp chat_template_issues("supported", []), do: []

  defp chat_template_issues("supported", missing_tokens) do
    ["chat template missing tokens: #{Enum.join(missing_tokens, ", ")}"]
  end

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

  defp tokenizer_token_types(metadata) do
    metadata
    |> metadata_array("tokenizer.ggml.token_type", [])
    |> Enum.frequencies_by(&token_type_name/1)
  end

  defp metadata_array(metadata, key, default) do
    case metadata_value(metadata, key) do
      %{values: values} -> values
      nil -> default
    end
  end

  defp token_type_name(1), do: "normal"
  defp token_type_name(2), do: "unknown"
  defp token_type_name(3), do: "control"
  defp token_type_name(4), do: "user_defined"
  defp token_type_name(5), do: "unused"
  defp token_type_name(6), do: "byte"
  defp token_type_name(_type_id), do: "undefined"

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
    |> put_special_flag(metadata, :add_bos, "tokenizer.ggml.add_bos_token")
    |> put_special_flag(metadata, :add_eos, "tokenizer.ggml.add_eos_token")
  end

  defp unsupported_features(metadata) do
    []
    |> add_sliding_window_issue(metadata)
    |> add_rope_scaling_issue(metadata)
    |> Enum.reverse()
  end

  defp unsupported_feature_metadata_values(metadata) do
    if unsupported_features(metadata) == [] do
      %{}
    else
      prefix = metadata_prefix(metadata)

      @unsupported_feature_detail_suffixes
      |> Enum.map(&metadata_key(prefix, &1))
      |> Enum.flat_map(fn key ->
        case metadata_value(metadata, key) do
          nil -> []
          "none" -> []
          value -> [{key, value}]
        end
      end)
      |> Map.new()
    end
  end

  defp add_sliding_window_issue(issues, metadata) do
    case metadata_value(
           metadata,
           metadata_key(metadata_prefix(metadata), "attention.sliding_window")
         ) do
      nil -> issues
      _window -> ["unsupported attention variant: sliding_window" | issues]
    end
  end

  defp add_rope_scaling_issue(issues, metadata) do
    case metadata_value(metadata, metadata_key(metadata_prefix(metadata), "rope.scaling.type")) do
      nil -> issues
      "none" -> issues
      type -> ["unsupported RoPE scaling: #{type}" | issues]
    end
  end

  defp put_special_token(attrs, metadata, tokens, name, key) do
    case metadata_value(metadata, key) do
      id when is_integer(id) ->
        Map.put(attrs, name, %{id: id, piece: Enum.at(tokens, id)})

      _other ->
        attrs
    end
  end

  defp put_special_flag(attrs, metadata, name, key) do
    case metadata_value(metadata, key) do
      value when is_boolean(value) -> Map.put(attrs, name, value)
      _other -> attrs
    end
  end

  defp metadata_value(metadata, key) do
    case Map.fetch(metadata, key) do
      {:ok, %{value: value}} -> value
      :error -> nil
    end
  end

  defp metadata_prefix(metadata) do
    Llamex.GGUF.ModelConfig.metadata_prefix(metadata)
  end

  defp metadata_key(prefix, suffix), do: "#{prefix}.#{suffix}"

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

  defp format_supported_combinations(combinations, opts \\ []) do
    combinations
    |> Enum.map(fn combination ->
      tokenizers = Enum.join(combination.tokenizers, "/")
      tokenizer_models = Enum.join(combination.tokenizer_models, "/")
      pre_tokenizers = Enum.join(combination.pre_tokenizers, "/")
      tensor_types = Enum.join(combination.tensor_types, "/")
      runtime_status = if opts[:runtime_status?], do: "+#{combination.runtime_status}", else: ""

      "#{combination.architecture}+#{tokenizers}+#{tokenizer_models}+#{pre_tokenizers}+#{tensor_types}#{runtime_status}"
    end)
    |> Enum.join("; ")
  end

  defp format_supported_tensor_type_ids(ids) do
    ids
    |> Enum.sort_by(fn {id, _name} -> id end)
    |> Enum.map_join(", ", fn {id, name} -> "#{id}:#{name}" end)
  end

  defp format_tensor_schema_surface(surface) do
    surface
    |> Enum.sort()
    |> Enum.map(fn {architecture, schema} ->
      unsupported = Enum.join(schema.unsupported_feature_parts, "/")
      unsupported = if unsupported == "", do: "none", else: unsupported

      "#{architecture}=interesting:#{length(schema.interesting_tensor_names)}, unsupported_features:#{unsupported}"
    end)
    |> Enum.join("; ")
  end

  defp format_architecture_runtime_surface(surface) do
    surface
    |> Enum.sort()
    |> Enum.map(fn {architecture, status} -> "#{architecture}=#{status}" end)
    |> Enum.join("; ")
  end

  defp format_tokenizer_metadata_surface(surface) do
    surface
    |> Enum.sort()
    |> Enum.map(fn {architecture, tokenizer} ->
      models = Enum.join(tokenizer.tokenizer_models, "/")
      pre_tokenizers = Enum.join(tokenizer.pre_tokenizers, "/")

      "#{architecture}=models:#{models}, pre:#{pre_tokenizers}"
    end)
    |> Enum.join("; ")
  end

  defp format_tokenizer_metadata(tokenizer) do
    models = Enum.join(tokenizer.tokenizer_models, "/")
    pre_tokenizers = Enum.join(tokenizer.pre_tokenizers, "/")

    "models:#{models}, pre:#{pre_tokenizers}"
  end

  defp format_model_config_surface(surface) do
    surface
    |> Enum.sort()
    |> Enum.map(fn {architecture, fields} ->
      names =
        fields
        |> Enum.map(& &1.name)
        |> Enum.join("/")

      "#{architecture}=#{names}"
    end)
    |> Enum.join("; ")
  end

  defp format_compatibility_issues([]), do: "none"

  defp format_compatibility_issues(issues), do: Enum.join(issues, "; ")

  defp format_compatibility_issue_groups(groups) do
    groups
    |> Enum.sort()
    |> Enum.map(fn {group, issues} ->
      issues = if issues == [], do: "none", else: Enum.join(issues, "; ")
      "#{group}=#{issues}"
    end)
    |> Enum.join(", ")
  end

  defp format_blocking_issue_groups([]), do: "none"
  defp format_blocking_issue_groups(groups), do: Enum.join(groups, ", ")

  defp format_missing_required_metadata([]), do: "none"

  defp format_missing_required_metadata(keys), do: Enum.join(keys, ", ")

  defp format_model_config(config) when map_size(config) == 0, do: "none"

  defp format_model_config(config) do
    config
    |> Enum.sort()
    |> Enum.map(fn {key, value} -> "#{key}=#{value}" end)
    |> Enum.join(", ")
  end

  defp format_missing_model_config_metadata([]), do: "none"

  defp format_missing_model_config_metadata(missing) do
    missing
    |> Enum.map(fn item -> "#{item.name}=#{item.metadata_key}" end)
    |> Enum.join(", ")
  end

  defp format_missing_required_tensors([]), do: "none"

  defp format_missing_required_tensors(names), do: Enum.join(names, ", ")

  defp format_tensor_shape_issues([]), do: "none"

  defp format_tensor_shape_issues(issues), do: Enum.join(issues, "; ")

  defp format_tensor_schema_mappings([]), do: "none"

  defp format_tensor_schema_mappings(mappings) do
    mappings
    |> Enum.map(fn mapping -> "#{mapping.name}->#{mapping.schema_name}" end)
    |> Enum.join(", ")
  end

  defp format_tensor_schema_issues([]), do: "none"

  defp format_tensor_schema_issues(issues), do: Enum.join(issues, "; ")

  defp format_chat_template_issues([]), do: "none"

  defp format_chat_template_issues(issues), do: Enum.join(issues, "; ")

  defp format_tokenizer_metadata_issues([]), do: "none"

  defp format_tokenizer_metadata_issues(issues), do: Enum.join(issues, "; ")

  defp format_special_tokens(tokens) when map_size(tokens) == 0, do: "none"

  defp format_special_tokens(tokens) do
    tokens
    |> Enum.sort()
    |> Enum.map(fn
      {name, %{id: id, piece: piece}} -> "#{name}=#{id}:#{piece}"
      {name, value} when is_boolean(value) -> "#{name}=#{value}"
    end)
    |> Enum.join(", ")
  end

  defp format_unsupported_features([]), do: "none"

  defp format_unsupported_features(features), do: Enum.join(features, "; ")

  defp format_metadata_values(values) when map_size(values) == 0, do: "none"

  defp format_metadata_values(values) do
    values
    |> Enum.sort()
    |> Enum.map(fn {key, value} -> "#{key}=#{value}" end)
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

  defp format_ratio(nil), do: "unknown"
  defp format_ratio(ratio), do: "#{Float.round(ratio, 2)}x"

  defp format_tensor_payload_by_type(payload) when map_size(payload) == 0, do: "none"

  defp format_tensor_payload_by_type(payload) do
    payload
    |> Enum.sort()
    |> Enum.map(fn {type_name, stats} ->
      "#{type_name}=tensors:#{stats.tensors}, elements:#{stats.elements}, gguf:#{format_bytes(stats.gguf_payload_bytes)}, eager_f32:#{format_bytes(stats.eager_f32_bytes)}, ratio:#{format_ratio(stats.eager_f32_expansion_ratio)}"
    end)
    |> Enum.join("; ")
  end

  defp format_top_tensor_payloads([]), do: "top tensor payloads: none"

  defp format_top_tensor_payloads(tensors) do
    tensors =
      tensors
      |> Enum.map(fn tensor ->
        "- #{tensor.name}: #{tensor.type} #{format_dimensions(tensor.dimensions)} elements:#{tensor.elements} gguf:#{format_bytes(tensor.gguf_payload_bytes)} eager_f32:#{format_bytes(tensor.eager_f32_bytes)} ratio:#{format_ratio(tensor.eager_f32_expansion_ratio)}"
      end)
      |> Enum.join("\n")

    "top tensor payloads:\n" <> tensors
  end

  defp format_tensor_shapes([]), do: "none"

  defp format_tensor_shapes(tensors) do
    tensors
    |> Enum.map(fn tensor ->
      "#{tensor.name}=#{tensor.type} gguf:#{format_dimensions(tensor.dimensions)} schema:#{format_dimensions(tensor.schema_shape)}"
    end)
    |> Enum.join(", ")
  end

  defp format_supported_tensors([]), do: "supported tensors: none"

  defp format_supported_tensors(tensors) do
    tensors =
      tensors
      |> Enum.map(fn tensor ->
        "- #{tensor.name}: #{tensor.type_name} #{format_dimensions(tensor.dimensions)}"
      end)
      |> Enum.join("\n")

    "supported tensors:\n" <> tensors
  end

  defp format_unsupported_tensors([]), do: ""

  defp format_unsupported_tensors(tensors) do
    tensors =
      tensors
      |> Enum.map(fn tensor ->
        "- #{tensor.name}: type_#{tensor.type} #{format_dimensions(tensor.dimensions)}"
      end)
      |> Enum.join("\n")

    "unsupported tensors:\n" <> tensors
  end

  defp format_dimensions(dimensions), do: inspect(dimensions, charlists: :as_lists)

  defp tensor_payload_bytes(%{type: 0, dimensions: dimensions}), do: element_count(dimensions) * 4
  defp tensor_payload_bytes(%{type: 1, dimensions: dimensions}), do: element_count(dimensions) * 2

  defp tensor_payload_bytes(%{type: 30, dimensions: dimensions}),
    do: element_count(dimensions) * 2

  defp tensor_payload_bytes(%{type: 2, dimensions: dimensions}),
    do: block_payload_bytes(dimensions, 32, 2 + div(32, 2))

  defp tensor_payload_bytes(%{type: 3, dimensions: dimensions}),
    do: block_payload_bytes(dimensions, 32, 4 + div(32, 2))

  defp tensor_payload_bytes(%{type: 6, dimensions: dimensions}),
    do: block_payload_bytes(dimensions, 32, 2 + 4 + div(32, 2))

  defp tensor_payload_bytes(%{type: 7, dimensions: dimensions}),
    do: block_payload_bytes(dimensions, 32, 4 + 4 + div(32, 2))

  defp tensor_payload_bytes(%{type: 8, dimensions: dimensions}),
    do: block_payload_bytes(dimensions, 32, 2 + 32)

  defp tensor_payload_bytes(%{type: 9, dimensions: dimensions}),
    do: block_payload_bytes(dimensions, 32, 4 + 32)

  defp tensor_payload_bytes(%{type: 10, dimensions: dimensions}),
    do: block_payload_bytes(dimensions, 256, 16 + 64 + 4)

  defp tensor_payload_bytes(%{type: 11, dimensions: dimensions}),
    do: block_payload_bytes(dimensions, 256, 32 + 64 + 12 + 2)

  defp tensor_payload_bytes(%{type: 12, dimensions: dimensions}),
    do: block_payload_bytes(dimensions, 256, 4 + 12 + div(256, 2))

  defp tensor_payload_bytes(%{type: 13, dimensions: dimensions}),
    do: block_payload_bytes(dimensions, 256, 4 + 12 + div(256, 8) + div(256, 2))

  defp tensor_payload_bytes(%{type: 14, dimensions: dimensions}),
    do: block_payload_bytes(dimensions, 256, 128 + 64 + 16 + 2)

  defp tensor_payload_bytes(%{type: 15, dimensions: dimensions}),
    do: block_payload_bytes(dimensions, 256, 4 + 256 + 32)

  defp tensor_payload_bytes(_tensor), do: 0

  defp block_payload_bytes(dimensions, block_size, bytes_per_block) do
    count = element_count(dimensions)

    if rem(count, block_size) == 0 do
      div(count, block_size) * bytes_per_block
    else
      0
    end
  end

  defp element_count(dimensions), do: Enum.product(dimensions)
end
