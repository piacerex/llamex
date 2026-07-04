defmodule Llamex.Tokenizer do
  @moduledoc """
  Tokenizer facade.
  """

  @type t :: Llamex.Tokenizer.Whitespace.t() | Llamex.Tokenizer.BPE.t()

  def new(vocab, unknown_token) when is_map(vocab) and is_binary(unknown_token) do
    whitespace(vocab, unknown_token)
  end

  def whitespace(vocab, unknown_token) when is_map(vocab) and is_binary(unknown_token) do
    Llamex.Tokenizer.Whitespace.new(vocab, unknown_token)
  end

  def bpe(vocab, merges, unknown_token)
      when is_map(vocab) and is_list(merges) and is_binary(unknown_token) do
    Llamex.Tokenizer.BPE.new(vocab, merges, unknown_token)
  end

  def encode(tokenizer, text) when is_binary(text) do
    tokenizer.__struct__.encode(tokenizer, text)
  end

  def decode(tokenizer, tokens) when is_list(tokens) do
    tokenizer.__struct__.decode(tokenizer, tokens)
  end
end
