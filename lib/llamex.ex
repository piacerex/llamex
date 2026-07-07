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

  def prefill(%Model{} = model, prompt, opts)
      when is_binary(prompt) and is_map(opts) do
    Generation.prefill(model, prompt, opts)
  end

  def prefill(%PreparedModel{} = prepared_model, prompt, opts)
      when is_binary(prompt) and is_map(opts) do
    Generation.prefill(prepared_model, prompt, opts)
  end

  def step(%Context{} = context, current_token, opts)
      when is_integer(current_token) and current_token >= 0 and is_map(opts) do
    Generation.step(context, current_token, opts)
  end
end
