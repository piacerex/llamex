defmodule Mix.Tasks.Llamex.Benchmark do
  @moduledoc """
  Runs generation benchmarks for one or more max token counts.

      mix llamex.benchmark model.gguf --json
      mix llamex.benchmark model.gguf --tokens 8,16,24,32 --backend nx_exla --exla cpu --natural
      mix llamex.benchmark model.gguf --backends list,nx_exla --tokens 8,16 --warmup 1 --repeat 3 --json
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
          backends: :string,
          exla: :string,
          json: :boolean,
          natural: :boolean,
          prompt: :string,
          tokens: :string,
          warmup: :integer,
          repeat: :integer,
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
    backends = backends(options)
    warmup_count = non_negative_integer_option(options, :warmup, 0)
    repeat_count = positive_integer_option(options, :repeat, 1)

    results =
      for backend <- backends, max_new_tokens <- token_counts do
        benchmark_case(
          model,
          model_path,
          prompt,
          max_new_tokens,
          backend,
          warmup_count,
          repeat_count,
          options
        )
      end

    print_results(results, options)
  end

  defp run_benchmark(_args, _options) do
    Mix.raise(~s(usage: mix llamex.benchmark MODEL [--tokens 8,16,24,32] [options]))
  end

  defp benchmark_case(
         model,
         model_path,
         prompt,
         max_new_tokens,
         backend,
         warmup_count,
         repeat_count,
         options
       ) do
    {prepare_microseconds, prepared_model} =
      :timer.tc(fn -> backend.prepare_model(model) end)

    warmups =
      Enum.map(1..warmup_count//1, fn index ->
        benchmark_once(
          prepared_model,
          model_path,
          prompt,
          max_new_tokens,
          backend,
          :warmup,
          index,
          options
        )
      end)

    runs =
      Enum.map(1..repeat_count, fn index ->
        benchmark_once(
          prepared_model,
          model_path,
          prompt,
          max_new_tokens,
          backend,
          :measured,
          index,
          options
        )
      end)

    %{
      model_path: model_path,
      backend: inspect(backend),
      prompt: prompt,
      requested_max_new_tokens: max_new_tokens,
      backend_prepare_milliseconds: div(prepare_microseconds, 1000),
      warmup_count: warmup_count,
      repeat_count: repeat_count,
      warmups: warmups,
      runs: runs,
      summary: summarize_runs(runs)
    }
  end

  defp benchmark_once(
         model,
         model_path,
         prompt,
         max_new_tokens,
         backend,
         phase,
         run_index,
         options
       ) do
    profile =
      Llamex.Profile.generation_steps(model, prompt, %{
        backend: backend,
        context_window: Map.get(options, :context_window),
        max_new_tokens: max_new_tokens,
        prepared_model: model,
        stop_tokens: stop_tokens(model, options),
        sampler: sampler(model, options)
      })

    text = maybe_trim_to_sentence(profile.text, options)
    generated_count = length(profile.generated_tokens)
    total_milliseconds = profile.timing_summary.total_milliseconds

    %{
      model_path: model_path,
      backend: inspect(backend),
      phase: phase,
      run_index: run_index,
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
      timing_components: profile.timing_summary.components,
      prompt_eval_steps: profile.prompt_eval_steps,
      prompt_eval_summary: profile.prompt_eval_summary
    }
  end

  defp print_results(results, %{json: true}) do
    Mix.shell().info(JSON.encode!(results))
  end

  defp print_results(results, _options) do
    Enum.each(results, fn result ->
      summary = result.summary

      Mix.shell().info(
        "backend=#{result.backend} tokens=#{result.requested_max_new_tokens} " <>
          "runs=#{result.repeat_count} warmups=#{result.warmup_count} " <>
          "backend_prepare_ms=#{result.backend_prepare_milliseconds} " <>
          "mean_ms=#{format_float(summary.total_milliseconds.mean)} " <>
          "median_ms=#{format_float(summary.total_milliseconds.median)} " <>
          "best_ms=#{format_float(summary.total_milliseconds.best)} " <>
          "tokens_per_second=#{format_float(summary.tokens_per_second.mean)} " <>
          "prompt_eval_top_layers=#{format_prompt_eval_top(result, :layers)} " <>
          "prompt_eval_top_components=#{format_prompt_eval_top(result, :components)}"
      )
    end)
  end

  defp token_counts(%{tokens: tokens}) do
    tokens
    |> String.split(",", trim: true)
    |> Enum.map(&parse_positive_integer!/1)
  end

  defp token_counts(_options), do: @default_tokens

  defp backends(%{backends: backends}) do
    backends
    |> String.split(",", trim: true)
    |> Enum.map(&backend(%{backend: String.trim(&1)}))
  end

  defp backends(%{backend: _backend} = options), do: [backend(options)]
  defp backends(options), do: [backend(options)]

  defp non_negative_integer_option(options, key, default) do
    case Map.get(options, key, default) do
      value when is_integer(value) and value >= 0 ->
        value

      value ->
        Mix.raise(
          "#{String.replace(to_string(key), "_", "-")} must be a non-negative integer, got: #{inspect(value)}"
        )
    end
  end

  defp positive_integer_option(options, key, default) do
    case Map.get(options, key, default) do
      value when is_integer(value) and value > 0 ->
        value

      value ->
        Mix.raise(
          "#{String.replace(to_string(key), "_", "-")} must be a positive integer, got: #{inspect(value)}"
        )
    end
  end

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
  defp format_float(value) when is_integer(value), do: Integer.to_string(value)
  defp format_float(value), do: :erlang.float_to_binary(value, decimals: 2)

  defp format_prompt_eval_top(result, key) do
    result.runs
    |> Enum.flat_map(fn run ->
      run.prompt_eval_summary
      |> Map.get(key, [])
      |> Enum.map(fn %{label: label, milliseconds: milliseconds} -> {label, milliseconds} end)
    end)
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
    |> Enum.map(fn {label, milliseconds} -> {label, Enum.sum(milliseconds)} end)
    |> Enum.sort_by(fn {_label, milliseconds} -> milliseconds end, :desc)
    |> Enum.take(3)
    |> case do
      [] ->
        "none"

      entries ->
        Enum.map_join(entries, ",", fn {label, milliseconds} ->
          "#{label}:#{format_float(milliseconds)}ms"
        end)
    end
  end

  defp summarize_runs(runs) do
    %{
      generated_tokens: numeric_summary(Enum.map(runs, & &1.generated_tokens)),
      total_milliseconds: numeric_summary(Enum.map(runs, & &1.total_milliseconds)),
      prefill_milliseconds: numeric_summary(Enum.map(runs, & &1.prefill_milliseconds)),
      step_milliseconds: numeric_summary(Enum.map(runs, & &1.step_milliseconds)),
      eval_milliseconds: numeric_summary(Enum.map(runs, & &1.eval_milliseconds)),
      milliseconds_per_generated_token:
        numeric_summary(Enum.map(runs, & &1.milliseconds_per_generated_token)),
      tokens_per_second: numeric_summary(Enum.map(runs, &tokens_per_second/1))
    }
  end

  defp tokens_per_second(%{total_milliseconds: total_milliseconds})
       when total_milliseconds in [0, nil],
       do: nil

  defp tokens_per_second(%{
         generated_tokens: generated_tokens,
         total_milliseconds: total_milliseconds
       }) do
    generated_tokens * 1000 / total_milliseconds
  end

  defp numeric_summary(values) do
    values = Enum.reject(values, &is_nil/1)

    if values == [] do
      %{best: nil, worst: nil, mean: nil, median: nil}
    else
      sorted = Enum.sort(values)

      %{
        best: List.first(sorted),
        worst: List.last(sorted),
        mean: Enum.sum(values) / length(values),
        median: median(sorted)
      }
    end
  end

  defp median(sorted) do
    count = length(sorted)
    middle = div(count, 2)

    if rem(count, 2) == 1 do
      Enum.at(sorted, middle)
    else
      (Enum.at(sorted, middle - 1) + Enum.at(sorted, middle)) / 2
    end
  end

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
