defmodule Mix.Tasks.Llamex.Gguf.Inspect do
  @moduledoc """
  Inspects GGUF model compatibility.

      mix llamex.gguf.inspect model.gguf
      mix llamex.gguf.inspect model.gguf --json
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
      OptionParser.parse(args, strict: [json: :boolean, schema: :boolean, supported: :boolean])

    if invalid != [] do
      Mix.raise("invalid options: #{inspect(invalid)}")
    end

    run_inspect(positional, Map.new(options))
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
      "usage: mix llamex.gguf.inspect MODEL_GGUF [MODEL_GGUF ...] [--json] [--schema] | --supported [--json]"
    )
  end

  defp tensor_schema_summary(path) do
    path
    |> Llamex.GGUF.Reader.read_metadata()
    |> Llamex.GGUF.ModelLoader.tensor_schema_summary()
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
end
