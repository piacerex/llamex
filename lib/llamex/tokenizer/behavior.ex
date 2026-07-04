defmodule Llamex.Tokenizer.Behavior do
  @moduledoc """
  Common tokenizer contract.
  """

  @callback encode(tokenizer :: term(), text :: String.t()) :: list(non_neg_integer())
  @callback decode(tokenizer :: term(), tokens :: list(non_neg_integer())) :: String.t()
end
