defmodule Llamex do
  @moduledoc """
  Minimal local LLM engine.

  `Llamex` is not an LLM API client. It is the public facade for constructing
  a tiny llama.cpp-inspired inference pipeline.
  """

  alias Llamex.{Config, Context, Engine, Generation, Model, Sampler}

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

  def eval(%Context{} = context, token) when is_integer(token) and token >= 0 do
    Engine.eval(context, token)
  end

  def next_token(%Context{} = context, token) when is_integer(token) and token >= 0 do
    Engine.next_token(context, token, &Sampler.greedy/2)
  end

  def encode(%Model{tokenizer: tokenizer}, text) when not is_nil(tokenizer) and is_binary(text) do
    Llamex.Tokenizer.encode(tokenizer, text)
  end

  def decode(%Model{tokenizer: tokenizer}, tokens)
      when not is_nil(tokenizer) and is_list(tokens) do
    Llamex.Tokenizer.decode(tokenizer, tokens)
  end

  def generate(%Model{} = model, prompt, opts)
      when is_binary(prompt) and is_map(opts) do
    Generation.generate(model, prompt, opts)
  end
end
