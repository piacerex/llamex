defmodule Mix.Tasks.Llamex.Gguf.Inspect do
  @moduledoc """
  Inspects GGUF model compatibility.

      mix llamex.gguf.inspect model.gguf
      mix llamex.gguf.inspect model.gguf --json
      mix llamex.gguf.inspect first.gguf second.gguf --json
  """

  use Mix.Task

  @shortdoc "Inspects GGUF model compatibility"

  @impl true
  def run(args) do
    {options, positional, invalid} = OptionParser.parse(args, strict: [json: :boolean])

    if invalid != [] do
      Mix.raise("invalid options: #{inspect(invalid)}")
    end

    run_inspect(positional, Map.new(options))
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
    Mix.raise("usage: mix llamex.gguf.inspect MODEL_GGUF [MODEL_GGUF ...] [--json]")
  end
end
