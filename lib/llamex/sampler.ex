defmodule Llamex.Sampler do
  @moduledoc """
  Token sampling strategies.
  """

  def greedy(logits, backend) when is_atom(backend) do
    backend.argmax(logits)
  end
end
