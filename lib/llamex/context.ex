defmodule Llamex.Context do
  @moduledoc """
  Runtime inference state.
  """

  @enforce_keys [:model, :backend, :tokens, :kv_cache]
  defstruct [:model, :backend, :tokens, :kv_cache]

  @type t :: %__MODULE__{
          model: Llamex.Model.t(),
          backend: module(),
          tokens: list(non_neg_integer()),
          kv_cache: Llamex.KVCache.t()
        }

  def new(model, backend) when is_atom(backend) do
    %__MODULE__{
      model: backend.prepare_model(model),
      backend: backend,
      tokens: [],
      kv_cache: Llamex.KVCache.new()
    }
  end

  def append(%__MODULE__{} = context, token) when is_integer(token) and token >= 0 do
    %{context | tokens: context.tokens ++ [token]}
  end
end
