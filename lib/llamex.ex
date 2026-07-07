defmodule Llamex do
  @moduledoc """
  Minimal local LLM engine.

  `Llamex` is not an LLM API client. It is the public facade for constructing
  a tiny llama.cpp-inspired inference pipeline.
  """

  alias Llamex.{Config, Context, Engine, Generation, Model, PreparedModel, Sampler}

  def new_model(attrs) when is_map(attrs) do
    config = Config.new(Map.fetch!(attrs, :config))

    Model.new(
      config,
      Map.fetch!(attrs, :token_embeddings),
      Map.drop(attrs, [:config, :token_embeddings])
    )
  end

  def new_context(%Model{} = model, backend) when is_atom(backend) do
    Context.new(model, backend)
  end

  def new_context(%PreparedModel{} = prepared_model) do
    Context.new_prepared(prepared_model.model, prepared_model.backend)
  end

  def prepare_model(%Model{} = model, backend) when is_atom(backend) do
    %PreparedModel{model: backend.prepare_model(model), backend: backend}
  end

  def eval(%Context{} = context, token) when is_integer(token) and token >= 0 do
    Engine.eval(context, token)
  end

  def next_token(%Context{} = context, token) when is_integer(token) and token >= 0 do
    Engine.next_token(context, token, &Sampler.greedy/2)
  end

  def encode(%Model{tokenizer: tokenizer}, text) when not is_nil(tokenizer) and is_binary(text) do
    Llamex.Tokenizer.encode(tokenizer, text)
  end

  def encode(%PreparedModel{} = prepared_model, text) when is_binary(text) do
    encode(prepared_model.model, text)
  end

  def decode(%Model{tokenizer: tokenizer}, tokens)
      when not is_nil(tokenizer) and is_list(tokens) do
    Llamex.Tokenizer.decode(tokenizer, tokens)
  end

  def decode(%PreparedModel{} = prepared_model, tokens) when is_list(tokens) do
    decode(prepared_model.model, tokens)
  end

  def generate(%Model{} = model, prompt, opts)
      when is_binary(prompt) and is_map(opts) do
    Generation.generate(model, prompt, opts)
  end

  def generate(%PreparedModel{} = prepared_model, prompt, opts)
      when is_binary(prompt) and is_map(opts) do
    Generation.generate(prepared_model, prompt, opts)
  end

  def generate_chat(model_or_prepared, messages, opts)
      when is_list(messages) and is_map(opts) do
    prompt = chat_prompt(model_or_prepared, messages, opts)
    generate(model_or_prepared, prompt, Map.delete(opts, :system))
  end

  def generate_chat(model_or_prepared, prompt, opts)
      when is_binary(prompt) and is_map(opts) do
    prompt = chat_prompt(model_or_prepared, prompt, opts)
    generate(model_or_prepared, prompt, Map.delete(opts, :system))
  end

  def prefill(%Model{} = model, prompt, opts)
      when is_binary(prompt) and is_map(opts) do
    Generation.prefill(model, prompt, opts)
  end

  def prefill(%PreparedModel{} = prepared_model, prompt, opts)
      when is_binary(prompt) and is_map(opts) do
    Generation.prefill(prepared_model, prompt, opts)
  end

  def chat_prompt(model_or_prepared, prompt_or_messages, opts \\ %{})

  def chat_prompt(%PreparedModel{} = prepared_model, prompt_or_messages, opts)
      when is_map(opts) do
    chat_prompt(prepared_model.model, prompt_or_messages, opts)
  end

  def chat_prompt(%Model{tokenizer: tokenizer} = model, prompt, opts)
      when not is_nil(tokenizer) and is_binary(prompt) and is_map(opts) do
    chat_prompt(model, chat_messages(prompt, opts), opts)
  end

  def chat_prompt(%Model{tokenizer: tokenizer}, messages, _opts)
      when not is_nil(tokenizer) and is_list(messages) do
    template =
      tokenizer.chat_template || raise ArgumentError, "model tokenizer has no chat template"

    if not Llamex.ChatTemplate.supported?(template) do
      raise ArgumentError, "unsupported chat template"
    end

    case Llamex.ChatTemplate.missing_tokens(template, tokenizer.token_to_id) do
      [] ->
        Llamex.ChatTemplate.apply(template, messages, tokenizer)

      missing ->
        raise ArgumentError,
              "chat template references missing tokenizer tokens: #{Enum.join(missing, ", ")}"
    end
  end

  defp chat_messages(prompt, %{system: system}) when is_binary(system) do
    [%{role: "system", content: system}, %{role: "user", content: prompt}]
  end

  defp chat_messages(prompt, _opts), do: [%{role: "user", content: prompt}]

  def step(%Context{} = context, current_token, opts)
      when is_integer(current_token) and current_token >= 0 and is_map(opts) do
    Generation.step(context, current_token, opts)
  end
end
