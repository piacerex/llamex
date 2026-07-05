defmodule Mix.Tasks.Llamex.Exla.Info do
  @moduledoc """
  Prints Nx/EXLA target information.

      mix llamex.exla.info
      mix llamex.exla.info --target cuda
      mix llamex.exla.info --target rocm --json
  """

  use Mix.Task

  @shortdoc "Prints Nx/EXLA target information"

  @impl true
  def run(args) do
    {options, _positional, invalid} =
      OptionParser.parse(args,
        strict: [target: :string, json: :boolean]
      )

    if invalid != [] do
      Mix.raise("invalid options: #{inspect(invalid)}")
    end

    Mix.Task.run("app.start")

    target = Keyword.get(options, :target, "cpu")
    info = Llamex.Backend.NxEXLA.info(target)

    if Keyword.get(options, :json, false) do
      Mix.shell().info(JSON.encode!(info))
    else
      Mix.shell().info(format(info))
    end
  rescue
    exception in [ArgumentError, RuntimeError] -> Mix.raise(Exception.message(exception))
  end

  defp format(info) do
    [
      "Nx available: #{info.nx_available?}",
      "EXLA available: #{info.exla_available?}",
      "target: #{info.target}",
      "client: #{info.client}",
      "XLA_TARGET: #{info.xla_target || "not set"}",
      "supported platforms: #{format_platforms(info.supported_platforms)}"
    ]
    |> Enum.join("\n")
  end

  defp format_platforms(platforms) when map_size(platforms) == 0, do: "unknown"

  defp format_platforms(platforms) do
    platforms
    |> Enum.sort_by(fn {platform, _count} -> platform end)
    |> Enum.map_join(", ", fn {platform, count} -> "#{platform}=#{count}" end)
  end
end
