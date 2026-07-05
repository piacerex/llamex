defmodule Mix.Tasks.Llamex.Natural.Baseline do
  @moduledoc """
  Runs the current natural-generation baseline gate.

      mix llamex.natural.baseline model.gguf
      mix llamex.natural.baseline model.gguf --json
      mix llamex.natural.baseline model.gguf --prompt "The quick brown fox"
  """

  use Mix.Task

  @shortdoc "Runs the natural generation baseline gate"

  @baseline_prompt "The quick brown fox"

  @impl true
  def run(args) do
    Mix.Tasks.Llamex.Natural.Smoke.run(baseline_args(args))
  end

  defp baseline_args(args) do
    args
    |> put_default_prompt()
    |> Kernel.++([
      "8",
      "--min-words",
      "4",
      "--reject-open-ending",
      "--complete-open-ending",
      "8",
      "--trim-to-sentence",
      "--fail-on-issue"
    ])
  end

  defp put_default_prompt(args) do
    if Enum.any?(args, &(&1 in ["--prompt", "-p"])) do
      args
    else
      args ++ ["--prompt", @baseline_prompt]
    end
  end
end
