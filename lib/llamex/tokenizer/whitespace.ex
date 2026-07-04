defmodule Llamex.Tokenizer.Whitespace do
  @moduledoc """
  Minimal whitespace tokenizer.

  This is intentionally not a BPE tokenizer yet. It provides the same engine
  boundary so a GGUF/BPE tokenizer can replace it later.
  """

  @behaviour Llamex.Tokenizer.Behavior

  @enforce_keys [:token_to_id, :id_to_token, :unknown_token]
  defstruct [:token_to_id, :id_to_token, :unknown_token, special_tokens: %{}, token_types: []]

  @type t :: %__MODULE__{
          token_to_id: %{required(String.t()) => non_neg_integer()},
          id_to_token: %{required(non_neg_integer()) => String.t()},
          unknown_token: String.t(),
          special_tokens: map(),
          token_types: list(map())
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
      token_types: Keyword.get(opts, :token_types, [])
    }
  end

  @impl true
  def encode(%__MODULE__{} = tokenizer, text) when is_binary(text) do
    text
    |> String.split()
    |> Enum.flat_map(&encode_token(tokenizer, &1))
  end

  @impl true
  def decode(%__MODULE__{} = tokenizer, tokens) when is_list(tokens) do
    Llamex.Tokenizer.ByteTokens.decode(tokenizer, tokens, fn tokenizer, tokens ->
      tokens
      |> Enum.map(&Map.fetch!(tokenizer.id_to_token, &1))
      |> Enum.join(" ")
    end)
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
