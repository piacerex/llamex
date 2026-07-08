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
    token_types: [],
    chat_template: nil
  ]

  @type t :: %__MODULE__{
          token_to_id: %{required(String.t()) => non_neg_integer()},
          id_to_token: %{required(non_neg_integer()) => String.t()},
          unknown_token: String.t(),
          merges: list({String.t(), String.t()}),
          special_tokens: map(),
          token_types: list(map()),
          chat_template: String.t() | nil
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
      tokenizer
      |> special_token_strings()
      |> Enum.sort_by(&byte_size/1, :desc)

    split_special_tokens(text, special_tokens, [])
  end

  defp special_token_strings(tokenizer) do
    marker_tokens =
      tokenizer.token_to_id
      |> Map.keys()
      |> Enum.filter(&String.starts_with?(&1, "<|"))

    configured_tokens =
      tokenizer.special_tokens
      |> Map.values()
      |> Enum.flat_map(fn
        %{token: token} when is_binary(token) -> [token]
        _other -> []
      end)

    Enum.uniq(marker_tokens ++ configured_tokens)
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
    word
    |> String.graphemes()
    |> apply_merges(tokenizer.merges)
    |> Enum.flat_map(&encode_token(tokenizer, &1))
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
