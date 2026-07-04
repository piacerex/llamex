defmodule Mix.Tasks.Llamex.Gguf.Inspect do
  @moduledoc """
  Inspects GGUF model compatibility.

      mix llamex.gguf.inspect model.gguf
  """

  use Mix.Task

  @shortdoc "Inspects GGUF model compatibility"

  @impl true
  def run([path]) do
    Mix.Task.run("app.start")

    path
    |> Llamex.GGUF.Diagnostic.inspect_file()
    |> Llamex.GGUF.Diagnostic.format()
    |> Mix.shell().info()
  end

  def run(_args) do
    Mix.raise("usage: mix llamex.gguf.inspect MODEL_GGUF")
  end
end
