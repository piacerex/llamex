defmodule Mix.Tasks.Llamex.Generate do
  @moduledoc """
  Generates text with a Llamex JSON model.

      mix llamex.generate priv/models/tiny.json hello 16
      mix llamex.generate priv/models/tiny.json hello 16 --temperature 0.8 --top-k 20 --top-p 0.9 --seed 42
  """

  use Mix.Task

  @shortdoc "Generates text with a Llamex JSON model"

  @impl true
  def run(args) do
    {options, positional, invalid} =
      OptionParser.parse(args,
        strict: [
          temperature: :float,
          top_k: :integer,
          top_p: :float,
          repetition_penalty: :float,
          seed: :integer
        ],
        aliases: [t: :temperature, k: :top_k, p: :top_p, s: :seed]
      )

    if invalid != [] do
      Mix.raise("invalid options: #{inspect(invalid)}")
    end

    run_generation(positional, Map.new(options))
  end

  defp run_generation([model_path, prompt], options) do
    run_generation([model_path, prompt, "16"], options)
  end

  defp run_generation([model_path, prompt, max_new_tokens], options) do
    Mix.Task.run("app.start")

    model = Llamex.ModelLoader.load_json(model_path)
    max_new_tokens = String.to_integer(max_new_tokens)

    result =
      Llamex.generate(model, prompt, %{
        backend: Llamex.Backend.List,
        max_new_tokens: max_new_tokens,
        stop_token: stop_token(model),
        sampler: sampler(options)
      })

    Mix.shell().info(result.text)
  end

  defp run_generation(_args, _options) do
    Mix.raise(~s(usage: mix llamex.generate MODEL_JSON "prompt text" [max_new_tokens] [options]))
  end

  defp sampler(options) when map_size(options) == 0, do: :greedy

  defp sampler(options) do
    options
    |> Map.put_new(:temperature, 1.0)
    |> Map.put_new(:seed, 1)
  end

  defp stop_token(%{tokenizer: nil}), do: nil

  defp stop_token(model) do
    model.tokenizer.token_to_id["<eos>"] || model.tokenizer.token_to_id["world"]
  end
end
