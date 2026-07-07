defmodule Mix.Tasks.Llamex.Benchmark do
  @moduledoc """
  Runs generation benchmarks for one or more max token counts.

      mix llamex.benchmark model.gguf --json
      mix llamex.benchmark model.gguf --tokens 8,16,24,32 --backend nx_exla --exla cpu --natural
      mix llamex.benchmark model.gguf --prompt "Elixir is" --tokens 24 --trim-to-sentence
  """

  use Mix.Task

  @shortdoc "Benchmarks generation across token counts"

  @default_prompt "The quick brown fox"
  @default_tokens [8, 16, 24, 32]

  @impl true
  def run(args) do
    {options, positional, invalid} =
      OptionParser.parse(args,
        strict: [
          backend: :string,
          exla: :string,
          json: :boolean,
          natural: :boolean,
          prompt: :string,
          tokens: :string,
          context_window: :integer,
          stop_control: :boolean,
          no_stop: :boolean,
          trim_to_sentence: :boolean,
          temperature: :float,
          top_k: :integer,
          top_p: :float,
          repetition_penalty: :float,
          no_repeat_ngram_size: :integer,
          no_repeat_adjacent_word: :boolean,
          seed: :integer
        ],
        aliases: [p: :prompt, t: :tokens, k: :top_k, s: :seed]
      )

    if invalid != [] do
      Mix.raise("invalid options: #{inspect(invalid)}")
    end

    run_benchmark(positional, Map.new(options))
  end

  defp run_benchmark([model_path], options) do
    Mix.Task.run("app.start")
    configure_exla(options)

    model = load_model(model_path)
    prompt = Map.get(options, :prompt, @default_prompt)
    token_counts = token_counts(options)

    results =
      Enum.map(token_counts, fn max_new_tokens ->
        benchmark_once(model, model_path, prompt, max_new_tokens, options)
      end)

    print_results(results, options)
  end

  defp run_benchmark(_args, _options) do
    Mix.raise(~s(usage: mix llamex.benchmark MODEL [--tokens 8,16,24,32] [options]))
  end

  defp benchmark_once(model, model_path, prompt, max_new_tokens, options) do
    profile =
      Llamex.Profile.generation_steps(model, prompt, %{
        backend: backend(options),
        context_window: Map.get(options, :context_window),
        max_new_tokens: max_new_tokens,
        stop_tokens: stop_tokens(model, options),
        sampler: sampler(model, options)
      })

    text = maybe_trim_to_sentence(profile.text, options)
    generated_count = length(profile.generated_tokens)
    total_milliseconds = profile.timing_summary.total_milliseconds

    %{
      model_path: model_path,
      backend: inspect(backend(options)),
      prompt: prompt,
      requested_max_new_tokens: max_new_tokens,
      generated_tokens: generated_count,
      finish_reason: profile.finish_reason,
      text: text,
      total_milliseconds: total_milliseconds,
      prefill_milliseconds: profile.timing_summary.prefill_milliseconds,
      step_milliseconds: profile.timing_summary.step_milliseconds,
      eval_milliseconds: profile.timing_summary.eval_milliseconds,
      milliseconds_per_generated_token:
        milliseconds_per_generated_token(total_milliseconds, generated_count),
      timing_components: profile.timing_summary.components
    }
  end

  defp print_results(results, %{json: true}) do
    Mix.shell().info(JSON.encode!(results))
  end

  defp print_results(results, _options) do
    Enum.each(results, fn result ->
      Mix.shell().info(
        [
          "tokens=#{result.requested_max_new_tokens}",
          "generated=#{result.generated_tokens}",
          "total_ms=#{result.total_milliseconds}",
          "ms_per_token=#{format_float(result.milliseconds_per_generated_token)}",
          "finish=#{result.finish_reason}",
          ~s(text=#{inspect(result.text)})
        ]
        |> Enum.join(" ")
      )
    end)
  end

  defp token_counts(%{tokens: tokens}) do
    tokens
    |> String.split(",", trim: true)
    |> Enum.map(&parse_positive_integer!/1)
  end

  defp token_counts(_options), do: @default_tokens

  defp parse_positive_integer!(value) do
    case Integer.parse(String.trim(value)) do
      {integer, ""} when integer > 0 ->
        integer

      _ ->
        Mix.raise(
          "tokens must be a comma-separated list of positive integers, got: #{inspect(value)}"
        )
    end
  end

  defp milliseconds_per_generated_token(_milliseconds, 0), do: nil

  defp milliseconds_per_generated_token(milliseconds, generated_tokens) do
    milliseconds / generated_tokens
  end

  defp format_float(nil), do: "n/a"
  defp format_float(value), do: :erlang.float_to_binary(value, decimals: 2)

  defp maybe_trim_to_sentence(text, %{trim_to_sentence: true}) do
    case Regex.run(~r/^(.+[.!?])([^.!?]*)$/us, text) do
      [_match, sentence, _discarded] -> sentence
      _ -> text
    end
  end

  defp maybe_trim_to_sentence(text, _options), do: text

  defp backend(%{backend: "list"}), do: Llamex.Backend.List
  defp backend(%{backend: "fpga"}), do: Llamex.Backend.FPGA
  defp backend(%{backend: "nx"}), do: Llamex.Backend.Nx
  defp backend(%{backend: "nx_exla"}), do: Llamex.Backend.NxEXLA
  defp backend(%{backend: nil}), do: Llamex.Backend.Nx
  defp backend(%{backend: backend}), do: Mix.raise("unsupported backend: #{backend}")
  defp backend(%{}), do: Llamex.Backend.Nx

  defp configure_exla(%{exla: target}) when is_binary(target) do
    Llamex.Backend.NxEXLA.configure!(target)
  rescue
    exception in [ArgumentError, RuntimeError] -> Mix.raise(Exception.message(exception))
  end

  defp configure_exla(_options), do: :ok

  defp stop_tokens(_model, %{no_stop: true}), do: []
  defp stop_tokens(model, %{stop_control: true}), do: Llamex.Natural.control_stop_tokens(model)
  defp stop_tokens(_model, _options), do: []

  defp sampler(model, %{natural: true} = options) do
    model
    |> Llamex.Natural.sampler(sampling_options(options))
  end

  defp sampler(_model, options) do
    sampling_options(options)
    |> sampler_from_options()
  end

  defp sampler_from_options(options) when map_size(options) == 0, do: :greedy

  defp sampler_from_options(options) do
    options
    |> Map.put_new(:temperature, 1.0)
    |> Map.put_new(:seed, 1)
  end

  defp sampling_options(options) do
    options
    |> Map.take([
      :temperature,
      :top_k,
      :top_p,
      :repetition_penalty,
      :no_repeat_ngram_size,
      :no_repeat_adjacent_word,
      :seed
    ])
  end

  defp load_model(model_path) do
    if Path.extname(model_path) == ".gguf" do
      Llamex.GGUF.ModelLoader.load(model_path)
    else
      Llamex.ModelLoader.load_json(model_path)
    end
  end
end
