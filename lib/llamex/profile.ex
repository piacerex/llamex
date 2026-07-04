defmodule Llamex.Profile do
  @moduledoc """
  Small profiling helpers for local GGUF generation experiments.
  """

  def timed(label, fun) when is_binary(label) and is_function(fun, 0) do
    {microseconds, result} = :timer.tc(fun)

    {%{label: label, milliseconds: div(microseconds, 1000)}, result}
  end

  def generation_step(model, prompt, opts) when is_binary(prompt) and is_map(opts) do
    backend = Map.get(opts, :backend, Llamex.Backend.List)
    sampler = Map.get(opts, :sampler, :greedy)

    {prefill_time, state} =
      timed("prefill", fn ->
        Llamex.prefill(model, prompt, %{backend: backend})
      end)

    {step_time, step} =
      timed("step", fn ->
        Llamex.step(state.context, state.current_token, %{sampler: sampler})
      end)

    %{
      prompt_tokens: length(state.prompt_tokens),
      token: step.token,
      text: step.text,
      timings: [prefill_time, step_time]
    }
  end
end
