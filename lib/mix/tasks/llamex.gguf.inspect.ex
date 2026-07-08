defmodule Mix.Tasks.Llamex.Gguf.Inspect do
  @moduledoc """
  Inspects GGUF model compatibility.

      mix llamex.gguf.inspect model.gguf
      mix llamex.gguf.inspect model.gguf --json
      mix llamex.gguf.inspect model.gguf --summary
      mix llamex.gguf.inspect first.gguf second.gguf --summary --json
      mix llamex.gguf.inspect model.gguf --config
      mix llamex.gguf.inspect first.gguf second.gguf --config --json
      mix llamex.gguf.inspect model.gguf --schema
      mix llamex.gguf.inspect first.gguf second.gguf --schema --json
      mix llamex.gguf.inspect first.gguf second.gguf --json
      mix llamex.gguf.inspect --supported
      mix llamex.gguf.inspect --supported --json
  """

  use Mix.Task

  @shortdoc "Inspects GGUF model compatibility"

  @impl true
  def run(args) do
    {options, positional, invalid} =
      OptionParser.parse(args,
        strict: [
          json: :boolean,
          summary: :boolean,
          config: :boolean,
          schema: :boolean,
          supported: :boolean
        ]
      )

    if invalid != [] do
      Mix.raise("invalid options: #{inspect(invalid)}")
    end

    run_inspect(positional, Map.new(options))
  end

  defp run_inspect(paths, %{summary: true, json: true}) when length(paths) > 0 do
    Mix.Task.run("app.start")

    summaries =
      Enum.map(paths, fn path ->
        path
        |> Llamex.GGUF.Diagnostic.inspect_summary_file()
        |> Map.put(:path, path)
      end)

    Mix.shell().info(JSON.encode!(summaries))
  end

  defp run_inspect([path], %{summary: true}) do
    Mix.Task.run("app.start")

    path
    |> Llamex.GGUF.Diagnostic.inspect_summary_file()
    |> format_summary()
    |> Mix.shell().info()
  end

  defp run_inspect(paths, %{config: true, json: true}) when length(paths) > 0 do
    Mix.Task.run("app.start")

    summaries =
      Enum.map(paths, fn path ->
        path
        |> model_config_report()
        |> Map.put(:path, path)
      end)

    Mix.shell().info(JSON.encode!(summaries))
  end

  defp run_inspect([path], %{config: true}) do
    Mix.Task.run("app.start")

    path
    |> model_config_report()
    |> format_model_config_report()
    |> Mix.shell().info()
  end

  defp run_inspect(paths, %{schema: true, json: true}) when length(paths) > 0 do
    Mix.Task.run("app.start")

    summaries =
      Enum.map(paths, fn path ->
        path
        |> tensor_schema_summary()
        |> Map.put(:path, path)
      end)

    Mix.shell().info(JSON.encode!(summaries))
  end

  defp run_inspect([path], %{schema: true}) do
    Mix.Task.run("app.start")

    path
    |> tensor_schema_summary()
    |> format_tensor_schema_summary()
    |> Mix.shell().info()
  end

  defp run_inspect([], %{supported: true, json: true}) do
    Mix.Task.run("app.start")

    Llamex.GGUF.Diagnostic.supported_surface()
    |> JSON.encode!()
    |> Mix.shell().info()
  end

  defp run_inspect([], %{supported: true}) do
    Mix.Task.run("app.start")

    Llamex.GGUF.Diagnostic.format_supported_surface()
    |> Mix.shell().info()
  end

  defp run_inspect(paths, %{json: true}) when length(paths) > 0 do
    Mix.Task.run("app.start")

    diagnostics =
      Enum.map(paths, fn path ->
        path
        |> Llamex.GGUF.Diagnostic.inspect_file()
        |> Map.put(:path, path)
      end)

    Mix.shell().info(JSON.encode!(diagnostics))
  end

  defp run_inspect([path], _options) do
    Mix.Task.run("app.start")

    path
    |> Llamex.GGUF.Diagnostic.inspect_file()
    |> Llamex.GGUF.Diagnostic.format()
    |> Mix.shell().info()
  end

  defp run_inspect(_args, _options) do
    Mix.raise(
      "usage: mix llamex.gguf.inspect MODEL_GGUF [MODEL_GGUF ...] [--json] [--summary] [--config] [--schema] | --supported [--json]"
    )
  end

  defp format_summary(summary) do
    [
      "architecture: #{summary.architecture || "unknown"}",
      "architecture runtime status: #{summary.architecture_runtime_status}",
      "architecture runtime blockers: #{format_list(summary.architecture_runtime_blockers)}",
      "architecture runtime blocker details: #{format_blocker_details(summary.architecture_runtime_blocker_details)}",
      "runtime feature status: #{format_runtime_feature_status(summary.runtime_feature_status)}",
      "runtime feature blockers: #{format_feature_blockers(summary.runtime_feature_blockers)}",
      "model combination: #{format_model_combination(summary.model_combination)}",
      "runtime capability: #{format_runtime_capability(summary.runtime_capability)}",
      "attention variant: #{format_variant(summary.attention_variant)}",
      "RoPE variant: #{format_variant(summary.rope_variant)}",
      "loadable: #{summary.loadable?}",
      "blocking issue groups: #{format_atoms(summary.blocking_issue_groups)}",
      "compatibility issues: #{format_list(summary.compatibility_issues)}",
      "compatibility issue groups: #{format_issue_groups(summary.compatibility_issue_groups)}",
      "tokenizer model: #{summary.tokenizer_model || "unknown"}",
      "tokenizer model supported: #{summary.tokenizer_model_supported?}",
      "pre-tokenizer: #{summary.pre_tokenizer || "unknown"}",
      "pre-tokenizer supported: #{summary.pre_tokenizer_supported?}",
      "tokenizer kind: #{summary.tokenizer_kind}",
      "tokenizer tokens: #{summary.tokenizer_token_count || "unknown"}",
      "tokenizer merges: #{summary.tokenizer_merge_count}",
      "tokenizer scores: #{summary.tokenizer_score_count}",
      "tokenizer token types: #{format_type_counts(summary.tokenizer_token_types)}",
      "special tokens: #{format_special_tokens(summary.special_tokens)}",
      "missing required metadata: #{format_list(summary.missing_required_metadata)}",
      "model config metadata prefix: #{summary.model_config_metadata_prefix}",
      "model config: #{format_model_config_summary(summary.model_config)}",
      "missing model config metadata: #{format_model_config_missing_metadata(summary.missing_model_config_metadata)}",
      "chat usable: #{summary.chat_usable}",
      "chat template: #{summary.chat_template}",
      "chat template family: #{summary.chat_template_family}",
      "chat template missing tokens: #{format_list(summary.missing_chat_template_tokens)}",
      "tokenizer metadata issues: #{format_list(summary.tokenizer_metadata_issues)}",
      "unsupported features: #{format_list(summary.unsupported_features)}",
      "unsupported feature metadata values: #{format_metadata_values(summary.unsupported_feature_metadata_values)}",
      "unsupported tensor features: #{format_list(summary.unsupported_tensor_features)}",
      "extra norm tensor layers: #{format_extra_norm_tensor_layers(summary.extra_norm_tensor_layers)}",
      "tensor schema mappings: #{format_mappings(summary.tensor_schema_mappings)}",
      "tensor schema issues: #{format_list(summary.tensor_schema_issues)}",
      "missing required tensors: #{format_list(summary.missing_required_tensors)}",
      "tensor shape issues: #{format_list(summary.tensor_shape_issues)}",
      "eager f32 lower bound: #{format_bytes(summary.eager_f32_bytes)}",
      "gguf payload bytes: #{format_bytes(summary.gguf_payload_bytes)}",
      "supported tensor types: #{format_type_counts(summary.supported_tensor_types)}",
      "unsupported tensor types: #{format_type_counts(summary.unsupported_tensor_types)}",
      "tensor payload by type: #{format_tensor_payload_by_type(summary.tensor_payload_by_type)}",
      format_top_tensor_payloads(summary.top_tensor_payloads)
    ]
    |> Enum.join("\n")
  end

  defp tensor_schema_summary(path) do
    Llamex.GGUF.ModelLoader.tensor_schema_summary_file(path)
  end

  defp model_config_report(path) do
    Llamex.GGUF.ModelLoader.model_config_report_file(path)
  end

  defp format_model_config_report(report) do
    [
      "metadata prefix: #{report["metadata_prefix"]}",
      "config: #{format_model_config_summary(report["config"])}",
      "missing metadata: #{format_model_config_missing_metadata(report["missing_metadata"])}"
    ]
    |> Enum.join("\n")
  end

  defp format_model_config_summary(config) do
    config
    |> Enum.sort_by(fn {key, _value} -> key end)
    |> Enum.map(fn {key, value} -> "#{key}: #{value}" end)
    |> Enum.join(", ")
  end

  defp format_model_config_missing_metadata([]), do: "none"

  defp format_model_config_missing_metadata(missing) do
    missing
    |> Enum.map(fn item -> "#{item.name}: #{item.metadata_key}" end)
    |> Enum.join(", ")
  end

  defp format_tensor_schema_summary(summary) do
    [
      "architecture: #{summary["architecture"] || "unknown"}",
      "mappings: #{format_mappings(summary["mappings"])}",
      "unsupported features: #{format_list(summary["unsupported_features"])}",
      "issues: #{format_list(summary["issues"])}"
    ]
    |> Enum.join("\n")
  end

  defp format_mappings([]), do: "none"

  defp format_mappings(mappings) do
    mappings
    |> Enum.map(fn mapping -> "#{mapping.name}->#{mapping.schema_name}" end)
    |> Enum.join(", ")
  end

  defp format_model_combination(combination) do
    tensor_types = format_list(combination.tensor_types)

    [
      "architecture=#{combination.architecture}",
      "runtime=#{combination.runtime_status}",
      "tokenizer=#{combination.tokenizer_kind}",
      "model=#{combination.tokenizer_model}",
      "pre=#{combination.pre_tokenizer}",
      "tensor_types=#{tensor_types}"
    ]
    |> Enum.join(", ")
  end

  defp format_runtime_capability(capability) do
    [
      "loadable=#{capability.loadable?}",
      "runtime=#{capability.runtime_status}",
      "runtime_blockers=#{format_list(capability.runtime_blockers)}",
      "runtime_blocker_details=#{format_blocker_details(capability.runtime_blocker_details)}",
      "feature_status=#{format_runtime_feature_status(capability.runtime_feature_status)}",
      "blocked_features=#{format_atoms(blocked_runtime_features(capability))}",
      "blocking_groups=#{format_atoms(capability.blocking_issue_groups)}",
      "attention=#{format_variant(capability.attention_variant)}",
      "rope=#{format_variant(capability.rope_variant)}"
    ]
    |> Enum.join(", ")
  end

  defp format_variant(%{} = variant) do
    variant
    |> Enum.sort()
    |> Enum.map_join(", ", fn {key, value} -> "#{key}=#{value}" end)
  end

  defp format_blocker_details([]), do: "none"

  defp format_blocker_details(details) do
    details
    |> Enum.map(fn detail -> "#{detail.id}:#{detail.component}:#{detail.reason}" end)
    |> Enum.join("/")
  end

  defp format_runtime_feature_status(statuses) do
    statuses
    |> Enum.sort()
    |> Enum.map_join(", ", fn {feature, status} -> "#{feature}=#{status}" end)
  end

  defp format_feature_blockers([]), do: "none"

  defp format_feature_blockers(blockers) do
    blockers
    |> Enum.map(fn blocker ->
      "#{blocker.feature}=#{blocker.component}:#{blocker.reason}"
    end)
    |> Enum.join(", ")
  end

  defp blocked_runtime_features(capability) do
    Map.get_lazy(capability, :blocked_runtime_features, fn ->
      capability.runtime_feature_status
      |> Enum.filter(fn {_feature, status} -> status == "blocked" end)
      |> Enum.map(fn {feature, _status} -> feature end)
      |> Enum.sort()
    end)
  end

  defp format_extra_norm_tensor_layers([]), do: "none"

  defp format_extra_norm_tensor_layers(tensors) do
    tensors
    |> Enum.map(fn tensor -> "blk.#{tensor.layer}.#{tensor.part}=#{tensor.name}" end)
    |> Enum.join(", ")
  end

  defp format_list([]), do: "none"
  defp format_list(values), do: Enum.join(values, "; ")

  defp format_metadata_values(values) when map_size(values) == 0, do: "none"

  defp format_metadata_values(values) do
    values
    |> Enum.sort()
    |> Enum.map(fn {key, value} -> "#{key}=#{value}" end)
    |> Enum.join(", ")
  end

  defp format_issue_groups(groups) do
    groups
    |> Enum.sort()
    |> Enum.map(fn {group, issues} ->
      issues = if issues == [], do: "none", else: Enum.join(issues, "; ")
      "#{group}=#{issues}"
    end)
    |> Enum.join(", ")
  end

  defp format_type_counts(counts) when map_size(counts) == 0, do: "none"

  defp format_type_counts(counts) do
    counts
    |> Enum.sort()
    |> Enum.map(fn {type, count} -> "#{type}=#{count}" end)
    |> Enum.join(", ")
  end

  defp format_special_tokens(tokens) when map_size(tokens) == 0, do: "none"

  defp format_special_tokens(tokens) do
    tokens
    |> Enum.sort()
    |> Enum.map(fn
      {name, %{} = token} -> "#{name}=#{token.id}:#{token.piece}"
      {name, value} when is_boolean(value) -> "#{name}=#{value}"
    end)
    |> Enum.join(", ")
  end

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
        "#{tensor.name}=#{tensor.type}, dims:#{format_dimensions(tensor.dimensions)}, elements:#{tensor.elements}, gguf:#{format_bytes(tensor.gguf_payload_bytes)}, eager_f32:#{format_bytes(tensor.eager_f32_bytes)}, ratio:#{format_ratio(tensor.eager_f32_expansion_ratio)}"
      end)
      |> Enum.join("; ")

    "top tensor payloads: " <> tensors
  end

  defp format_dimensions(dimensions), do: Enum.join(dimensions, "x")

  defp format_atoms([]), do: "none"

  defp format_atoms(values) do
    values
    |> Enum.map(&Atom.to_string/1)
    |> Enum.join(", ")
  end

  defp format_bytes(nil), do: "unknown"
  defp format_bytes(bytes), do: "#{bytes} B"

  defp format_ratio(nil), do: "unknown"
  defp format_ratio(ratio), do: "#{Float.round(ratio, 2)}x"
end
