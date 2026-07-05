defmodule Mix.Tasks.Llamex.Natural.Smoke do
  @moduledoc """
  Runs a small natural-generation smoke suite against a model.

      mix llamex.natural.smoke model.gguf
      mix llamex.natural.smoke model.gguf 3 --json
      mix llamex.natural.smoke model.gguf 3 --json --fail-on-issue
      mix llamex.natural.smoke model.gguf 3 --json --min-words 2
      mix llamex.natural.smoke model.gguf 8 --json --reject-open-ending
      mix llamex.natural.smoke model.gguf 3 --prompt "Elixir is"
  """

  use Mix.Task

  @shortdoc "Runs natural generation smoke prompts"

  @default_prompts ["Elixir is", "Once upon a time", "The quick brown fox"]

  @impl true
  def run(args) do
    {options, positional, invalid} =
      OptionParser.parse(args,
        strict: [
          backend: :string,
          fail_on_issue: :boolean,
          json: :boolean,
          reject_open_ending: :boolean,
          prompt: :keep,
          min_words: :integer,
          max_new_tokens: :integer,
          seed: :integer,
          top_p: :float,
          top_k: :integer,
          temperature: :float,
          repetition_penalty: :float,
          no_repeat_ngram_size: :integer
        ],
        aliases: [p: :prompt, s: :seed]
      )

    if invalid != [] do
      Mix.raise("invalid options: #{inspect(invalid)}")
    end

    run_smoke(positional, Map.new(options))
  end

  defp run_smoke([model_path], options), do: run_smoke([model_path, "3"], options)

  defp run_smoke([model_path, max_new_tokens], options) do
    Mix.Task.run("app.start")

    model = load_model(model_path)
    max_new_tokens = Map.get(options, :max_new_tokens, String.to_integer(max_new_tokens))
    prompts = prompts(options)
    sampler = natural_sampler(model, options)
    stop_tokens = Llamex.Natural.control_stop_tokens(model)

    results =
      Enum.map(prompts, fn prompt ->
        result =
          Llamex.generate(model, prompt, %{
            backend: backend(options),
            max_new_tokens: max_new_tokens,
            stop_tokens: stop_tokens,
            sampler: sampler
          })

        check =
          Llamex.Natural.smoke_check(model, result.generated_tokens, result.text, %{
            finish_reason: result.finish_reason,
            min_words: min_words(options),
            reject_open_ending: Map.get(options, :reject_open_ending, false)
          })

        %{
          prompt: prompt,
          text: result.text,
          generated_tokens: result.generated_tokens,
          finish_reason: result.finish_reason,
          ok: check.ok,
          issues: check.issues
        }
      end)

    print_results(results, options)
    maybe_fail_on_issue(results, options)
  end

  defp run_smoke(_args, _options) do
    Mix.raise(~s(usage: mix llamex.natural.smoke MODEL [max_new_tokens] [--json] [--prompt TEXT]))
  end

  defp prompts(%{prompt: prompts}) when is_list(prompts), do: prompts
  defp prompts(%{prompt: prompt}) when is_binary(prompt), do: [prompt]
  defp prompts(_options), do: @default_prompts

  defp min_words(%{min_words: min_words}) when is_integer(min_words) and min_words > 0,
    do: min_words

  defp min_words(%{min_words: min_words}),
    do: Mix.raise("min_words must be a positive integer, got: #{inspect(min_words)}")

  defp min_words(_options), do: 1

  defp natural_sampler(model, options) do
    sampler_options =
      Map.take(options, [:temperature, :top_k, :top_p, :repetition_penalty, :seed])
      |> Map.merge(Map.take(options, [:no_repeat_ngram_size]))

    Llamex.Natural.sampler(model, sampler_options)
  end

  defp backend(%{backend: "list"}), do: Llamex.Backend.List
  defp backend(%{backend: "nx"}), do: Llamex.Backend.Nx
  defp backend(%{backend: nil}), do: Llamex.Backend.List
  defp backend(%{backend: backend}), do: Mix.raise("unsupported backend: #{backend}")
  defp backend(%{}), do: Llamex.Backend.List

  defp load_model(model_path) do
    if Path.extname(model_path) == ".gguf" do
      Llamex.GGUF.ModelLoader.load(model_path)
    else
      Llamex.ModelLoader.load_json(model_path)
    end
  end

  defp print_results(results, %{json: true}) do
    Mix.shell().info(JSON.encode!(results))
  end

  defp print_results(results, _options) do
    Enum.each(results, fn result ->
      status = if result.ok, do: "ok", else: "issue"
      Mix.shell().info("[#{status}] #{result.prompt} => #{result.text}")
    end)
  end

  defp maybe_fail_on_issue(results, %{fail_on_issue: true}) do
    failed = Enum.reject(results, & &1.ok)

    if failed != [] do
      Mix.raise("natural smoke issues: #{length(failed)} prompt(s) failed")
    end
  end

  defp maybe_fail_on_issue(_results, _options), do: :ok
end
