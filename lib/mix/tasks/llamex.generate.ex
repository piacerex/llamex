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
          backend: :string,
          exla: :string,
          temperature: :float,
          top_k: :integer,
          top_p: :float,
          repetition_penalty: :float,
          no_repeat_ngram_size: :integer,
          no_repeat_adjacent_word: :boolean,
          seed: :integer,
          chat: :boolean,
          system: :string,
          natural: :boolean,
          profile: :boolean,
          context_window: :integer,
          candidates: :integer,
          stop_token: :integer,
          stop_piece: :string,
          stop_sequence: :string,
          stop_special: :string,
          stop_control: :boolean,
          no_stop: :boolean
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
    configure_exla(options)

    validate_chat_template(model_path, options)

    model = load_model(model_path)
    max_new_tokens = String.to_integer(max_new_tokens)

    original_prompt = prompt
    prompt = maybe_apply_chat_template(model, prompt, options)

    run_model(model, model_path, original_prompt, prompt, max_new_tokens, options)
  end

  defp run_generation(_args, _options) do
    Mix.raise(~s(usage: mix llamex.generate MODEL_JSON "prompt text" [max_new_tokens] [options]))
  end

  defp sampler(%{natural: true} = options) do
    options
    |> sampling_options()
  end

  defp sampler(options) do
    options = sampling_options(options)

    if map_size(options) == 0 do
      :greedy
    else
      options
      |> Map.put_new(:temperature, 1.0)
      |> Map.put_new(:seed, 1)
    end
  end

  defp sampling_options(options) do
    options
    |> Map.delete(:backend)
    |> Map.delete(:exla)
    |> Map.delete(:natural)
    |> Map.delete(:chat)
    |> Map.delete(:system)
    |> Map.delete(:profile)
    |> Map.delete(:context_window)
    |> Map.delete(:candidates)
    |> Map.delete(:stop_token)
    |> Map.delete(:stop_piece)
    |> Map.delete(:stop_sequence)
    |> Map.delete(:stop_special)
    |> Map.delete(:stop_control)
    |> Map.delete(:no_stop)
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

  defp stop_token(_model, %{no_stop: true}), do: nil

  defp stop_token(_model, %{stop_token: stop_token}), do: stop_token

  defp stop_token(model, %{stop_piece: piece}) do
    tokenizer = model.tokenizer || Mix.raise("--stop-piece requires a model tokenizer")

    Map.get(tokenizer.token_to_id, piece) ||
      Mix.raise("stop piece not found in tokenizer vocab: #{piece}")
  end

  defp stop_token(model, %{stop_special: name}) do
    tokenizer = model.tokenizer || Mix.raise("--stop-special requires a model tokenizer")
    key = special_token_key(name)

    get_in(tokenizer.special_tokens, [key, :id]) ||
      Mix.raise("special stop token not found: #{name}")
  end

  defp stop_token(%{tokenizer: nil}, _options), do: nil

  defp stop_token(model, _options) do
    get_in(model.tokenizer.special_tokens, [:eos, :id]) ||
      model.tokenizer.token_to_id["<eos>"] ||
      model.tokenizer.token_to_id["</s>"] ||
      model.tokenizer.token_to_id["world"]
  end

  defp stop_tokens(_model, %{no_stop: true}), do: []

  defp stop_tokens(model, options) do
    model
    |> explicit_stop_tokens(options)
    |> Kernel.++(control_stop_tokens(model, options))
    |> Enum.uniq()
  end

  defp explicit_stop_tokens(model, options) do
    case stop_token(model, options) do
      nil -> []
      stop_token -> [stop_token]
    end
  end

  defp control_stop_tokens(%{tokenizer: nil}, _options), do: []

  defp control_stop_tokens(model, %{stop_control: true}) do
    Llamex.Natural.control_stop_tokens(model)
  end

  defp control_stop_tokens(_model, _options), do: []

  defp stop_sequences(%{stop_sequence: stop_sequence}) when is_binary(stop_sequence) do
    if stop_sequence == "", do: [], else: [stop_sequence]
  end

  defp stop_sequences(_options), do: []

  defp special_token_key("unknown"), do: :unknown
  defp special_token_key("bos"), do: :bos
  defp special_token_key("eos"), do: :eos
  defp special_token_key("padding"), do: :padding
  defp special_token_key(name), do: Mix.raise("unsupported special stop token: #{name}")

  defp run_model(
         model,
         model_path,
         original_prompt,
         prompt,
         max_new_tokens,
         %{profile: true} = options
       ) do
    profile =
      Llamex.Profile.generation_steps(model, prompt, %{
        backend: backend(options),
        context_window: Map.get(options, :context_window),
        max_new_tokens: max_new_tokens,
        stop_tokens: stop_tokens(model, options),
        stop_sequences: stop_sequences(options),
        sampler: sampler(model, options),
        candidate_count: Map.get(options, :candidates, 0)
      })
      |> Map.put(:model_path, model_path)
      |> Map.put(:original_prompt, original_prompt)
      |> Map.put(:prompt, prompt)

    Mix.shell().info(JSON.encode!(profile))
  end

  defp run_model(model, _model_path, _original_prompt, prompt, max_new_tokens, options) do
    result =
      Llamex.generate(model, prompt, %{
        backend: backend(options),
        context_window: Map.get(options, :context_window),
        max_new_tokens: max_new_tokens,
        stop_tokens: stop_tokens(model, options),
        stop_sequences: stop_sequences(options),
        sampler: sampler(model, options)
      })

    Mix.shell().info(result.text)
  end

  defp sampler(model, options) do
    sampler = sampler(options)

    if Map.get(options, :natural) == true and is_map(sampler) do
      Llamex.Natural.sampler(model, sampler)
    else
      sampler
    end
  end

  defp load_model(model_path) do
    if Path.extname(model_path) == ".gguf" do
      Llamex.GGUF.ModelLoader.load(model_path)
    else
      Llamex.ModelLoader.load_json(model_path)
    end
  end

  defp validate_chat_template(model_path, %{chat: true}) do
    if Path.extname(model_path) == ".gguf" do
      model_path
      |> Llamex.GGUF.Reader.read_metadata()
      |> Map.fetch!(:metadata)
      |> Llamex.GGUF.Tokenizer.from_metadata()
      |> validate_chat_tokenizer!()
    end
  end

  defp validate_chat_template(_model_path, _options), do: :ok

  defp maybe_apply_chat_template(model, prompt, %{chat: true} = options) do
    tokenizer = model.tokenizer || Mix.raise("--chat requires a model tokenizer")
    validate_chat_tokenizer!(tokenizer)

    Llamex.ChatTemplate.apply(tokenizer.chat_template, chat_messages(prompt, options), tokenizer)
  end

  defp maybe_apply_chat_template(_model, prompt, _options), do: prompt

  defp chat_messages(prompt, %{system: system}) when is_binary(system) do
    [%{role: "system", content: system}, %{role: "user", content: prompt}]
  end

  defp chat_messages(prompt, _options), do: [%{role: "user", content: prompt}]

  defp validate_chat_tokenizer!(tokenizer) do
    template = tokenizer.chat_template || Mix.raise("--chat requires tokenizer.chat_template")

    if not Llamex.ChatTemplate.supported?(template) do
      Mix.raise("unsupported chat template; run mix llamex.gguf.inspect MODEL_GGUF first")
    end

    case Llamex.ChatTemplate.missing_tokens(template, tokenizer.token_to_id) do
      [] ->
        :ok

      missing ->
        Mix.raise(
          "chat template references missing tokenizer tokens: #{Enum.join(missing, ", ")}"
        )
    end
  end
end
