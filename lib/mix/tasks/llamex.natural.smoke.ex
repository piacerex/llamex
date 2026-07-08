defmodule Mix.Tasks.Llamex.Natural.Smoke do
  @moduledoc """
  Runs a small natural-generation smoke suite against a model.

      mix llamex.natural.smoke model.gguf
      mix llamex.natural.smoke model.gguf 3 --json
      mix llamex.natural.smoke model.gguf 3 --json --fail-on-issue
      mix llamex.natural.smoke model.gguf 3 --json --min-words 2
      mix llamex.natural.smoke model.gguf 8 --json --reject-open-ending
      mix llamex.natural.smoke model.gguf 8 --json --complete-open-ending 4
      mix llamex.natural.smoke model.gguf 8 --json --trim-to-sentence
      mix llamex.natural.smoke model.gguf 3 --json --include-japanese
      mix llamex.natural.smoke model.gguf 3 --prompt "Elixir is"
  """

  use Mix.Task

  @shortdoc "Runs natural generation smoke prompts"

  @default_prompts ["Elixir is", "Once upon a time", "The quick brown fox"]
  @japanese_prompts ["こんにちは"]

  @impl true
  def run(args) do
    {options, positional, invalid} =
      OptionParser.parse(args,
        strict: [
          backend: :string,
          exla: :string,
          fail_on_issue: :boolean,
          json: :boolean,
          reject_open_ending: :boolean,
          complete_open_ending: :integer,
          trim_to_sentence: :boolean,
          include_japanese: :boolean,
          prompt: :keep,
          min_words: :integer,
          max_new_tokens: :integer,
          context_window: :integer,
          seed: :integer,
          top_p: :float,
          min_p: :float,
          top_k: :integer,
          temperature: :float,
          repetition_penalty: :float,
          no_repeat_ngram_size: :integer,
          no_repeat_adjacent_word: :boolean
        ],
        aliases: [p: :prompt, s: :seed]
      )

    if invalid != [] do
      Mix.raise("invalid options: #{inspect(invalid)}")
    end

    run_smoke(positional, options_map(options))
  end

  defp run_smoke([model_path], options), do: run_smoke([model_path, "3"], options)

  defp run_smoke([model_path, max_new_tokens], options) do
    Mix.Task.run("app.start")
    configure_exla(options)

    model = load_model(model_path)
    model_diagnostic = model_diagnostic(model_path)
    max_new_tokens = Map.get(options, :max_new_tokens, String.to_integer(max_new_tokens))
    prompts = prompts(options)
    backend = backend(options)
    sampler = natural_sampler(model, options)
    stop_tokens = Llamex.Natural.control_stop_tokens(model)
    settings = smoke_settings(max_new_tokens, stop_tokens, sampler, backend, options)

    results =
      smoke_results!(
        model,
        model_path,
        model_diagnostic,
        prompts,
        max_new_tokens,
        stop_tokens,
        sampler,
        backend,
        settings,
        options
      )

    print_results(results, options)
    maybe_fail_on_issue(results, options)
  rescue
    exception in ArgumentError -> Mix.raise(Exception.message(exception))
  end

  defp run_smoke(_args, _options) do
    Mix.raise(~s(usage: mix llamex.natural.smoke MODEL [max_new_tokens] [--json] [--prompt TEXT]))
  end

  defp options_map(options) do
    options
    |> Keyword.delete(:prompt)
    |> Map.new()
    |> put_prompts(Keyword.get_values(options, :prompt))
  end

  defp put_prompts(options, []), do: options
  defp put_prompts(options, [prompt]), do: Map.put(options, :prompt, prompt)
  defp put_prompts(options, prompts), do: Map.put(options, :prompt, prompts)

  defp smoke_results!(
         model,
         model_path,
         model_diagnostic,
         prompts,
         max_new_tokens,
         stop_tokens,
         sampler,
         backend,
         settings,
         options
       ) do
    Enum.map(prompts, fn prompt ->
      result =
        Llamex.generate(model, prompt, %{
          backend: backend,
          context_window: Map.get(options, :context_window),
          max_new_tokens: max_new_tokens,
          stop_tokens: stop_tokens,
          sampler: sampler
        })

      result =
        maybe_complete_open_ending(
          model,
          prompt,
          result,
          sampler,
          stop_tokens,
          backend,
          max_new_tokens,
          options
        )
        |> maybe_trim_to_sentence(options)

      check =
        Llamex.Natural.smoke_check(model, result.generated_tokens, result.text, %{
          finish_reason: result.finish_reason,
          min_words: min_words(options),
          reject_open_ending: Map.get(options, :reject_open_ending, false)
        })

      %{
        model_path: model_path,
        model_diagnostic: model_diagnostic,
        prompt: prompt,
        settings: settings,
        text: result.text,
        prompt_tokens: result.prompt_tokens,
        prompt_pieces: result.prompt_pieces,
        generated_tokens: result.generated_tokens,
        generated_pieces: token_pieces(model, result.generated_tokens),
        completion_tokens: result.completion_tokens,
        completion_pieces: token_pieces(model, result.completion_tokens),
        discarded_text: Map.get(result, :discarded_text, ""),
        finish_reason: result.finish_reason,
        ok: check.ok,
        issues: check.issues
      }
    end)
  rescue
    exception in ArgumentError -> Mix.raise(Exception.message(exception))
  end

  defp prompts(%{prompt: prompts}) when is_list(prompts), do: prompts
  defp prompts(%{prompt: prompt}) when is_binary(prompt), do: [prompt]
  defp prompts(%{include_japanese: true}), do: @default_prompts ++ @japanese_prompts
  defp prompts(_options), do: @default_prompts

  defp min_words(%{min_words: min_words}) when is_integer(min_words) and min_words > 0,
    do: min_words

  defp min_words(%{min_words: min_words}),
    do: Mix.raise("min_words must be a positive integer, got: #{inspect(min_words)}")

  defp min_words(_options), do: 1

  defp smoke_settings(max_new_tokens, stop_tokens, sampler, backend, options) do
    %{
      backend: inspect(backend),
      exla: exla_settings(backend),
      max_new_tokens: max_new_tokens,
      min_words: min_words(options),
      reject_open_ending: Map.get(options, :reject_open_ending, false),
      complete_open_ending: complete_open_ending(options),
      trim_to_sentence: Map.get(options, :trim_to_sentence, false),
      context_window: Map.get(options, :context_window),
      stop_tokens: stop_tokens,
      sampler: display_sampler(sampler)
    }
  end

  defp display_sampler(%{suppress_tokens: suppress_tokens} = sampler)
       when is_list(suppress_tokens) do
    sampler
    |> Map.delete(:suppress_tokens)
    |> Map.put(:suppressed_token_count, length(suppress_tokens))
  end

  defp display_sampler(sampler), do: sampler

  defp token_pieces(%{tokenizer: %{id_to_token: id_to_token}}, token_ids) do
    Enum.map(token_ids, &Map.get(id_to_token, &1, Integer.to_string(&1)))
  end

  defp token_pieces(_model, token_ids), do: Enum.map(token_ids, &Integer.to_string/1)

  defp exla_settings(backend) when backend in [Llamex.Backend.Nx, Llamex.Backend.NxEXLA] do
    Llamex.Backend.NxEXLA.configured()
  end

  defp exla_settings(_backend), do: nil

  defp maybe_complete_open_ending(
         model,
         prompt,
         result,
         sampler,
         stop_tokens,
         backend,
         _max_new_tokens,
         options
       ) do
    extra_tokens = complete_open_ending(options)

    if extra_tokens > 0 and result.finish_reason == :length and
         Llamex.Natural.open_ending?(result.text) do
      complete_open_ending(
        model,
        prompt,
        result,
        sampler,
        stop_tokens,
        backend,
        options,
        extra_tokens,
        []
      )
    else
      Map.put(result, :completion_tokens, [])
    end
  end

  defp complete_open_ending(
         _model,
         _prompt,
         result,
         _sampler,
         _stop_tokens,
         _backend,
         _options,
         0,
         tokens
       ) do
    Map.put(result, :completion_tokens, Enum.reverse(tokens))
  end

  defp complete_open_ending(
         model,
         prompt,
         result,
         sampler,
         stop_tokens,
         backend,
         options,
         remaining,
         tokens
       ) do
    chunk_tokens = min(4, remaining)

    completion =
      Llamex.generate(model, continuation_prompt(prompt, result.text), %{
        backend: backend,
        context_window: Map.get(options, :context_window),
        max_new_tokens: chunk_tokens,
        stop_tokens: stop_tokens,
        sampler: sampler
      })

    result = %{
      result
      | text: append_completion_text(result.text, completion.text),
        generated_tokens: result.generated_tokens ++ completion.generated_tokens,
        finish_reason: completion.finish_reason
    }

    tokens = Enum.reverse(completion.generated_tokens) ++ tokens
    remaining = remaining - length(completion.generated_tokens)

    if remaining > 0 and result.finish_reason == :length and
         Llamex.Natural.open_ending?(result.text) do
      complete_open_ending(
        model,
        prompt,
        result,
        sampler,
        stop_tokens,
        backend,
        options,
        remaining,
        tokens
      )
    else
      Map.put(result, :completion_tokens, Enum.reverse(tokens))
    end
  end

  defp complete_open_ending(%{complete_open_ending: tokens})
       when is_integer(tokens) and tokens >= 0,
       do: tokens

  defp complete_open_ending(%{complete_open_ending: tokens}),
    do: Mix.raise("complete_open_ending must be a non-negative integer, got: #{inspect(tokens)}")

  defp complete_open_ending(_options), do: 0

  defp maybe_trim_to_sentence(result, %{trim_to_sentence: true}) do
    case trim_to_sentence(result.text) do
      {text, discarded_text} when discarded_text != "" ->
        %{result | text: text, finish_reason: :trimmed}
        |> Map.put(:discarded_text, discarded_text)

      _ ->
        Map.put_new(result, :discarded_text, "")
    end
  end

  defp maybe_trim_to_sentence(result, _options), do: Map.put_new(result, :discarded_text, "")

  defp trim_to_sentence(text) do
    case Regex.run(~r/^(.+[.!?])([^.!?]*)$/us, text) do
      [_match, sentence, discarded] -> {sentence, discarded}
      _ -> {text, ""}
    end
  end

  defp continuation_prompt(prompt, generated_text) do
    join_text(prompt, generated_text)
  end

  defp append_completion_text(text, completion_text) do
    if Regex.match?(~r/[[:alnum:],;:]$/u, text) and
         Regex.match?(~r/^[[:alnum:]]/u, completion_text) do
      text <> " " <> completion_text
    else
      text <> completion_text
    end
  end

  defp join_text(left, right) do
    if Regex.match?(~r/\s$/u, left) or Regex.match?(~r/^\s/u, right) do
      left <> right
    else
      left <> " " <> right
    end
  end

  defp natural_sampler(model, options) do
    sampler_options =
      Map.take(options, [:temperature, :top_k, :top_p, :min_p, :repetition_penalty, :seed])
      |> Map.merge(Map.take(options, [:no_repeat_ngram_size, :no_repeat_adjacent_word]))
      |> Map.delete(:exla)

    Llamex.Natural.sampler(model, sampler_options)
  end

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

  defp load_model(model_path) do
    if Path.extname(model_path) == ".gguf" do
      Llamex.GGUF.ModelLoader.load(model_path)
    else
      Llamex.ModelLoader.load_json(model_path)
    end
  end

  defp model_diagnostic(model_path) do
    if Path.extname(model_path) == ".gguf" do
      Llamex.GGUF.Diagnostic.inspect_summary_file(model_path)
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
