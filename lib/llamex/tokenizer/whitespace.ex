defmodule Llamex.Tokenizer.Whitespace do
  @moduledoc """
  Minimal whitespace tokenizer.

  This is intentionally not a BPE tokenizer yet. It provides the same engine
  boundary so a GGUF/BPE tokenizer can replace it later.
  """

  @behaviour Llamex.Tokenizer.Behavior

  @enforce_keys [:token_to_id, :id_to_token, :unknown_token]
  defstruct [
    :token_to_id,
    :id_to_token,
    :unknown_token,
    special_tokens: %{},
    token_types: [],
    chat_template: nil
  ]

  @type t :: %__MODULE__{
          token_to_id: %{required(String.t()) => non_neg_integer()},
          id_to_token: %{required(non_neg_integer()) => String.t()},
          unknown_token: String.t(),
          special_tokens: map(),
          token_types: list(map()),
          chat_template: String.t() | nil
        }

  def new(vocab, unknown_token) when is_map(vocab) and is_binary(unknown_token) do
    new(vocab, unknown_token, [])
  end

  def new(vocab, unknown_token, opts) when is_map(vocab) and is_binary(unknown_token) do
    if not Map.has_key?(vocab, unknown_token) do
      raise ArgumentError, "vocab must contain unknown_token"
    end

    %__MODULE__{
      token_to_id: vocab,
      id_to_token: Map.new(vocab, fn {token, id} -> {id, token} end),
      unknown_token: unknown_token,
      special_tokens: Keyword.get(opts, :special_tokens, %{}),
      token_types: Keyword.get(opts, :token_types, []),
      chat_template: Keyword.get(opts, :chat_template)
    }
  end

  @impl true
  def encode(%__MODULE__{} = tokenizer, text) when is_binary(text) do
    tokenizer
    |> split_special_tokens(text)
    |> Enum.flat_map(fn
      {:special, token} -> [Map.fetch!(tokenizer.token_to_id, token)]
      {:text, text} -> text |> String.split() |> Enum.flat_map(&encode_word(tokenizer, &1))
    end)
    |> add_configured_special_tokens(tokenizer)
  end

  @impl true
  def decode(%__MODULE__{} = tokenizer, tokens) when is_list(tokens) do
    Llamex.Tokenizer.ByteTokens.decode(tokenizer, tokens, fn tokenizer, tokens ->
      Llamex.Tokenizer.TextDecoder.decode(tokenizer, tokens)
    end)
  end

  defp split_special_tokens(tokenizer, text) do
    special_tokens =
      tokenizer.token_to_id
      |> Map.keys()
      |> Enum.filter(&String.starts_with?(&1, "<|"))
      |> Enum.sort_by(&byte_size/1, :desc)

    split_special_tokens(text, special_tokens, [])
  end

  defp split_special_tokens("", _special_tokens, parts), do: Enum.reverse(parts)

  defp split_special_tokens(text, special_tokens, parts) do
    case Enum.find(special_tokens, &String.starts_with?(text, &1)) do
      nil ->
        {plain, rest} = take_until_special(text, special_tokens, "")
        split_special_tokens(rest, special_tokens, [{:text, plain} | parts])

      token ->
        rest = binary_part(text, byte_size(token), byte_size(text) - byte_size(token))
        split_special_tokens(rest, special_tokens, [{:special, token} | parts])
    end
  end

  defp take_until_special("", _special_tokens, acc), do: {acc, ""}

  defp take_until_special(text, special_tokens, acc) do
    if Enum.any?(special_tokens, &String.starts_with?(text, &1)) do
      {acc, text}
    else
      <<char::utf8, rest::binary>> = text
      take_until_special(rest, special_tokens, acc <> <<char::utf8>>)
    end
  end

  defp encode_word(tokenizer, word) do
    sentencepiece_token = "▁" <> word

    cond do
      Map.has_key?(tokenizer.token_to_id, sentencepiece_token) ->
        [Map.fetch!(tokenizer.token_to_id, sentencepiece_token)]

      sentencepiece_vocab?(tokenizer) ->
        encode_sentencepiece_word(tokenizer, sentencepiece_token, word)

      true ->
        encode_token(tokenizer, word)
    end
  end

  defp sentencepiece_vocab?(tokenizer) do
    Enum.any?(Map.keys(tokenizer.token_to_id), &String.starts_with?(&1, "▁"))
  end

  defp encode_sentencepiece_word(tokenizer, sentencepiece_token, fallback_word) do
    case encode_longest_pieces(tokenizer, sentencepiece_token, []) do
      {:ok, token_ids} -> token_ids
      :error -> encode_token(tokenizer, fallback_word)
    end
  end

  defp encode_longest_pieces(_tokenizer, "", token_ids), do: {:ok, Enum.reverse(token_ids)}

  defp encode_longest_pieces(tokenizer, text, token_ids) do
    case longest_piece(tokenizer, text) do
      nil ->
        :error

      {piece, token_id} ->
        rest = binary_part(text, byte_size(piece), byte_size(text) - byte_size(piece))
        encode_longest_pieces(tokenizer, rest, [token_id | token_ids])
    end
  end

  defp longest_piece(tokenizer, text) do
    tokenizer.token_to_id
    |> Map.keys()
    |> Enum.filter(&String.starts_with?(text, &1))
    |> Enum.max_by(&byte_size/1, fn -> nil end)
    |> case do
      nil -> nil
      piece -> {piece, Map.fetch!(tokenizer.token_to_id, piece)}
    end
  end

  defp encode_token(tokenizer, token) do
    case Map.fetch(tokenizer.token_to_id, token) do
      {:ok, id} ->
        [id]

      :error ->
        case Llamex.Tokenizer.ByteTokens.encode(tokenizer, token) do
          :error -> [Map.fetch!(tokenizer.token_to_id, tokenizer.unknown_token)]
          byte_tokens -> byte_tokens
        end
    end
  end

  defp add_configured_special_tokens(token_ids, tokenizer) do
    token_ids
    |> maybe_prepend_special(tokenizer.special_tokens[:add_bos], tokenizer.special_tokens[:bos])
    |> maybe_append_special(tokenizer.special_tokens[:add_eos], tokenizer.special_tokens[:eos])
  end

  defp maybe_prepend_special(token_ids, true, %{id: id}) do
    case token_ids do
      [^id | _rest] -> token_ids
      _other -> [id | token_ids]
    end
  end

  defp maybe_prepend_special(token_ids, _enabled, _special), do: token_ids

  defp maybe_append_special(token_ids, true, %{id: id}) do
    case Enum.reverse(token_ids) do
      [^id | _rest] -> token_ids
      _other -> token_ids ++ [id]
    end
  end

  defp maybe_append_special(token_ids, _enabled, _special), do: token_ids
end
