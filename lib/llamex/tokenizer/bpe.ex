defmodule Llamex.Tokenizer.BPE do
  @moduledoc """
  Minimal BPE tokenizer.

  This is a small fixture-oriented BPE implementation. It is not byte-level BPE
  yet, but it establishes the shape needed for tokenizer.json and GGUF metadata.
  """

  @behaviour Llamex.Tokenizer.Behavior

  @enforce_keys [:token_to_id, :id_to_token, :unknown_token, :merges]
  defstruct [
    :token_to_id,
    :id_to_token,
    :unknown_token,
    :merges,
    special_tokens: %{},
    token_types: []
  ]

  @type t :: %__MODULE__{
          token_to_id: %{required(String.t()) => non_neg_integer()},
          id_to_token: %{required(non_neg_integer()) => String.t()},
          unknown_token: String.t(),
          merges: list({String.t(), String.t()}),
          special_tokens: map(),
          token_types: list(map())
        }

  def new(vocab, merges, unknown_token)
      when is_map(vocab) and is_list(merges) and is_binary(unknown_token) do
    new(vocab, merges, unknown_token, [])
  end

  def new(vocab, merges, unknown_token, opts)
      when is_map(vocab) and is_list(merges) and is_binary(unknown_token) do
    if not Map.has_key?(vocab, unknown_token) do
      raise ArgumentError, "vocab must contain unknown_token"
    end

    %__MODULE__{
      token_to_id: vocab,
      id_to_token: Map.new(vocab, fn {token, id} -> {id, token} end),
      unknown_token: unknown_token,
      merges: Enum.map(merges, &parse_merge/1),
      special_tokens: Keyword.get(opts, :special_tokens, %{}),
      token_types: Keyword.get(opts, :token_types, [])
    }
  end

  @impl true
  def encode(%__MODULE__{} = tokenizer, text) when is_binary(text) do
    text
    |> String.split()
    |> Enum.flat_map(&encode_word(tokenizer, &1))
  end

  @impl true
  def decode(%__MODULE__{} = tokenizer, tokens) when is_list(tokens) do
    if has_byte_tokens?(tokenizer, tokens) do
      tokens
      |> Enum.map(&decode_token(tokenizer, &1))
      |> IO.iodata_to_binary()
    else
      tokens
      |> Enum.map(&Map.fetch!(tokenizer.id_to_token, &1))
      |> Enum.join(" ")
    end
  end

  defp encode_word(tokenizer, word) do
    word
    |> String.graphemes()
    |> apply_merges(tokenizer.merges)
    |> Enum.map(fn token ->
      Map.get(
        tokenizer.token_to_id,
        token,
        Map.fetch!(tokenizer.token_to_id, tokenizer.unknown_token)
      )
    end)
  end

  defp apply_merges(tokens, merges) do
    Enum.reduce(merges, tokens, fn {left, right}, tokens ->
      merge_pair(tokens, left, right)
    end)
  end

  defp merge_pair([left, right | rest], left, right) do
    [left <> right | merge_pair(rest, left, right)]
  end

  defp merge_pair([token | rest], left, right), do: [token | merge_pair(rest, left, right)]
  defp merge_pair([], _left, _right), do: []

  defp parse_merge([left, right]) when is_binary(left) and is_binary(right), do: {left, right}

  defp parse_merge(merge) when is_binary(merge) do
    case String.split(merge) do
      [left, right] -> {left, right}
      _other -> raise ArgumentError, "merge strings must contain exactly two tokens"
    end
  end

  defp has_byte_tokens?(tokenizer, tokens) do
    byte_token_ids = byte_token_ids(tokenizer)

    Enum.any?(tokens, &MapSet.member?(byte_token_ids, &1))
  end

  defp decode_token(tokenizer, id) do
    if MapSet.member?(byte_token_ids(tokenizer), id) do
      <<byte_value(Map.fetch!(tokenizer.id_to_token, id))>>
    else
      Map.fetch!(tokenizer.id_to_token, id)
    end
  end

  defp byte_token_ids(tokenizer) do
    tokenizer.token_types
    |> Enum.filter(&(&1.type == :byte))
    |> Enum.map(& &1.id)
    |> MapSet.new()
  end

  defp byte_value("<0x" <> rest) do
    rest
    |> String.trim_trailing(">")
    |> String.to_integer(16)
  end
end
