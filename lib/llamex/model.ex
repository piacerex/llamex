defmodule Llamex.Model do
  @moduledoc """
  Minimal model container.

  The first implementation is intentionally tiny: token embeddings are projected
  back to vocabulary logits by dot product. It is enough to exercise the same
  loading, evaluation, and sampling boundaries used by larger llama.cpp-style
  engines.
  """

  alias Llamex.Config

  @enforce_keys [:config, :token_embeddings]
  defstruct [:config, :token_embeddings, :tokenizer, :layers, :output_norm, :output]

  @type t :: %__MODULE__{
          config: Config.t(),
          token_embeddings: %{required(non_neg_integer()) => list(number())},
          tokenizer: Llamex.Tokenizer.t() | nil,
          layers: list(map()) | nil,
          output_norm: list(number()) | nil,
          output: map() | nil
        }

  def new(%Config{} = config, token_embeddings) when is_map(token_embeddings) do
    new(config, token_embeddings, %{})
  end

  def new(%Config{} = config, token_embeddings, attrs)
      when is_map(token_embeddings) and is_map(attrs) do
    expected_tokens = MapSet.new(0..(config.vocab_size - 1))
    actual_tokens = token_embeddings |> Map.keys() |> MapSet.new()

    if not MapSet.equal?(expected_tokens, actual_tokens) do
      raise ArgumentError, "token_embeddings must contain exactly vocab_size entries"
    end

    Enum.each(token_embeddings, fn {_token, embedding} ->
      if length(embedding) != config.embedding_size do
        raise ArgumentError, "each token embedding must match embedding_size"
      end
    end)

    %__MODULE__{
      config: config,
      token_embeddings: token_embeddings,
      tokenizer: Map.get(attrs, :tokenizer),
      layers: Map.get(attrs, :layers, []),
      output_norm: Map.get(attrs, :output_norm),
      output: Map.get(attrs, :output)
    }
  end
end
