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
      {:text, text} -> text |> String.split() |> Enum.flat_map(&encode_token(tokenizer, &1))
    end)
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
end
