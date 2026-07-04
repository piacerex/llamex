defmodule Llamex.Config do
  @moduledoc """
  Static model configuration.
  """

  @enforce_keys [:vocab_size, :embedding_size]
  defstruct [:vocab_size, :embedding_size, :context_size, :epsilon, :rope_theta]

  @type t :: %__MODULE__{
          vocab_size: pos_integer(),
          embedding_size: pos_integer(),
          context_size: pos_integer() | nil,
          epsilon: number() | nil,
          rope_theta: number() | nil
        }

  def new(attrs) when is_map(attrs) do
    vocab_size = Map.fetch!(attrs, :vocab_size)
    embedding_size = Map.fetch!(attrs, :embedding_size)

    if vocab_size <= 0 or embedding_size <= 0 do
      raise ArgumentError, "vocab_size and embedding_size must be positive"
    end

    %__MODULE__{
      vocab_size: vocab_size,
      embedding_size: embedding_size,
      context_size: Map.get(attrs, :context_size),
      epsilon: Map.get(attrs, :epsilon, 1.0e-6),
      rope_theta: Map.get(attrs, :rope_theta, 10_000.0)
    }
  end
end
