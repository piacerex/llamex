defmodule Mix.Tasks.Llamex.Gguf.Inspect do
  @moduledoc """
  Inspects GGUF model compatibility.

      mix llamex.gguf.inspect model.gguf
      mix llamex.gguf.inspect model.gguf --json
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

  defp run_inspect([path], options) do
    Mix.Task.run("app.start")

    diagnostic = Llamex.GGUF.Diagnostic.inspect_file(path)

    if Map.get(options, :json, false) do
      Mix.shell().info(JSON.encode!(diagnostic))
    else
      diagnostic
      |> Llamex.GGUF.Diagnostic.format()
      |> Mix.shell().info()
    end
  end

  defp run_inspect(_args, _options) do
    Mix.raise("usage: mix llamex.gguf.inspect MODEL_GGUF [--json]")
  end
end
