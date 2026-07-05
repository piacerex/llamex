defmodule Mix.Tasks.Llamex.Natural.Baseline do
  @moduledoc """
  Runs the current natural-generation baseline gate.

      mix llamex.natural.baseline model.gguf
      mix llamex.natural.baseline model.gguf --json
      mix llamex.natural.baseline model.gguf --prompt "The quick brown fox"
  """

  use Mix.Task

  @shortdoc "Runs the natural generation baseline gate"

  @impl true
  def run(args) do
    Mix.Tasks.Llamex.Natural.Smoke.run(baseline_args(args))
  end

  defp baseline_args(args) do
    args ++
      [
        "8",
        "--min-words",
        "4",
        "--reject-open-ending",
        "--complete-open-ending",
        "4",
        "--fail-on-issue"
      ]
  end
end
