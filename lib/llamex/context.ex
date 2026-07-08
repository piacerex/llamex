defmodule Llamex.Context do
  @moduledoc """
  Runtime inference state.
  """

  @enforce_keys [:model, :backend, :tokens, :token_count, :kv_cache]
  defstruct [:model, :backend, :tokens, :token_count, :kv_cache]

  @type t :: %__MODULE__{
          model: Llamex.Model.t(),
          backend: module(),
          tokens: list(non_neg_integer()),
          token_count: non_neg_integer(),
          kv_cache: Llamex.KVCache.t()
        }

  def new(model, backend) when is_atom(backend) do
    Llamex.RuntimeCapability.validate!(model)

    %__MODULE__{
      model: backend.prepare_model(model),
      backend: backend,
      tokens: [],
      token_count: 0,
      kv_cache: Llamex.KVCache.new()
    }
  end

  def new_prepared(model, backend) when is_atom(backend) do
    Llamex.RuntimeCapability.validate!(model)

    %__MODULE__{
      model: model,
      backend: backend,
      tokens: [],
      token_count: 0,
      kv_cache: Llamex.KVCache.new()
    }
  end

  def append(%__MODULE__{} = context, token) when is_integer(token) and token >= 0 do
    %{context | tokens: context.tokens ++ [token], token_count: context.token_count + 1}
  end
end
