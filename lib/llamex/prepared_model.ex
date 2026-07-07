defmodule Llamex.PreparedModel do
  @moduledoc """
  Backend-prepared model wrapper.

  Use this when the same loaded model is reused for multiple prompts with the
  same backend. It keeps one-time backend tensor preparation outside each
  generation call.
  """

  @enforce_keys [:model, :backend]
  defstruct [:model, :backend]

  @type t :: %__MODULE__{
          model: Llamex.Model.t(),
          backend: module()
        }
end
