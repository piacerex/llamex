defmodule Llamex.Config do
  @moduledoc """
  Static model configuration.
  """

  @enforce_keys [:vocab_size, :embedding_size]
  defstruct [
    :vocab_size,
    :embedding_size,
    :context_size,
    :epsilon,
    :rope_theta,
    :rope_dimension_count,
    :block_count,
    :attention_head_count,
    :attention_head_count_kv,
    :attention_sliding_window,
    :feed_forward_size
  ]

  @type t :: %__MODULE__{
          vocab_size: pos_integer(),
          embedding_size: pos_integer(),
          context_size: pos_integer() | nil,
          epsilon: number() | nil,
          rope_theta: number() | nil,
          rope_dimension_count: pos_integer() | nil,
          block_count: pos_integer() | nil,
          attention_head_count: pos_integer() | nil,
          attention_head_count_kv: pos_integer() | nil,
          attention_sliding_window: pos_integer() | nil,
          feed_forward_size: pos_integer() | nil
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
      rope_theta: Map.get(attrs, :rope_theta, 10_000.0),
      rope_dimension_count: Map.get(attrs, :rope_dimension_count),
      block_count: Map.get(attrs, :block_count),
      attention_head_count: Map.get(attrs, :attention_head_count),
      attention_head_count_kv: Map.get(attrs, :attention_head_count_kv),
      attention_sliding_window: Map.get(attrs, :attention_sliding_window),
      feed_forward_size: Map.get(attrs, :feed_forward_size)
    }
  end
end
