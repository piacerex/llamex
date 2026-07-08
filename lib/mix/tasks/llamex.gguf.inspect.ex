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
      "loadable: #{summary.loadable?}",
      "blocking issue groups: #{format_atoms(summary.blocking_issue_groups)}",
      "compatibility issues: #{format_list(summary.compatibility_issues)}",
      "chat usable: #{summary.chat_usable}",
      "chat template family: #{summary.chat_template_family}",
      "tokenizer metadata issues: #{format_list(summary.tokenizer_metadata_issues)}",
      "unsupported tensor features: #{format_list(summary.unsupported_tensor_features)}",
      "tensor schema issues: #{format_list(summary.tensor_schema_issues)}",
      "eager f32 lower bound: #{format_bytes(summary.eager_f32_bytes)}",
      "gguf payload bytes: #{format_bytes(summary.gguf_payload_bytes)}"
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

  defp format_list([]), do: "none"
  defp format_list(values), do: Enum.join(values, "; ")

  defp format_atoms([]), do: "none"

  defp format_atoms(values) do
    values
    |> Enum.map(&Atom.to_string/1)
    |> Enum.join(", ")
  end

  defp format_bytes(nil), do: "unknown"
  defp format_bytes(bytes), do: "#{bytes} B"
end
