defmodule Mix.Tasks.Llamex.Tokenize do
  @moduledoc """
  Tokenizes a prompt with a Llamex model tokenizer.

      mix llamex.tokenize priv/models/tiny.json hello
      mix llamex.tokenize model.gguf "Hello" --chat
  """

  use Mix.Task

  @shortdoc "Tokenizes a prompt with a Llamex model tokenizer"

  @impl true
  def run(args) do
    {options, positional, invalid} =
      OptionParser.parse(args,
        strict: [chat: :boolean, system: :string]
      )

    if invalid != [] do
      Mix.raise("invalid options: #{inspect(invalid)}")
    end

    run_tokenize(positional, Map.new(options))
  end

  defp run_tokenize([model_path, prompt], options) do
    Mix.Task.run("app.start")

    tokenizer = load_tokenizer(model_path)
    prompt = maybe_apply_chat_template(tokenizer, prompt, options)
    token_ids = Llamex.Tokenizer.encode(tokenizer, prompt)

    result = %{
      prompt: prompt,
      token_count: length(token_ids),
      tokens: Enum.map(token_ids, &token_info(tokenizer, &1))
    }

    Mix.shell().info(JSON.encode!(result))
  end

  defp run_tokenize(_args, _options) do
    Mix.raise(~s(usage: mix llamex.tokenize MODEL "prompt text" [--chat]))
  end

  defp load_tokenizer(model_path) do
    if Path.extname(model_path) == ".gguf" do
      model_path
      |> Llamex.GGUF.Reader.read_metadata()
      |> Map.fetch!(:metadata)
      |> Llamex.GGUF.Tokenizer.from_metadata()
    else
      model_path
      |> Llamex.ModelLoader.load_json()
      |> Map.fetch!(:tokenizer)
    end
  end

  defp maybe_apply_chat_template(tokenizer, prompt, %{chat: true} = options) do
    validate_chat_tokenizer!(tokenizer)

    Llamex.ChatTemplate.apply(tokenizer.chat_template, chat_messages(prompt, options), tokenizer)
  end

  defp maybe_apply_chat_template(_tokenizer, prompt, _options), do: prompt

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
          "chat template references missing tokenizer tokens: #{Enum.join(missing, ", ")}; run mix llamex.gguf.inspect MODEL_GGUF first"
        )
    end
  end

  defp token_info(tokenizer, id) do
    tokenizer
    |> token_type(id)
    |> Map.merge(%{
      id: id,
      piece: Map.fetch!(tokenizer.id_to_token, id)
    })
  end

  defp token_type(tokenizer, id) do
    case Enum.find(tokenizer.token_types, &(&1.id == id)) do
      nil -> %{}
      %{type: type, type_id: type_id} -> %{type: type, type_id: type_id}
    end
  end
end
