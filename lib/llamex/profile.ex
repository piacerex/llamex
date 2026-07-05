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

  def prefill_steps(model, prompt, opts) when is_binary(prompt) and is_map(opts) do
    backend = Map.get(opts, :backend, Llamex.Backend.List)
    prompt_tokens = Llamex.encode(model, prompt)
    context = Llamex.Context.new(model, backend)
    prefill_tokens = Enum.drop(prompt_tokens, -1)

    {steps, context} =
      prefill_tokens
      |> Enum.with_index(1)
      |> Enum.reduce({[], context}, fn {token, index}, {steps, context} ->
        {timing, {context, _logits}} =
          timed("prefill_#{index}", fn ->
            Llamex.Engine.eval(context, token)
          end)

        step = %{
          index: index,
          token: token,
          piece: Map.fetch!(model.tokenizer.id_to_token, token),
          timing: timing
        }

        {[step | steps], context}
      end)

    %{
      prompt_tokens: prompt_tokens,
      current_token: List.last(prompt_tokens),
      current_piece: Map.fetch!(model.tokenizer.id_to_token, List.last(prompt_tokens)),
      context_tokens: context.tokens,
      steps: Enum.reverse(steps)
    }
  end

  def generation_steps(model, prompt, opts) when is_binary(prompt) and is_map(opts) do
    backend = Map.get(opts, :backend, Llamex.Backend.List)
    sampler = Map.get(opts, :sampler, :greedy)
    max_new_tokens = Map.get(opts, :max_new_tokens, 1)

    {prefill_time, state} =
      timed("prefill", fn ->
        Llamex.prefill(model, prompt, %{backend: backend})
      end)

    {steps, _context, _current_token, _sampler_state} =
      Enum.reduce(1..max_new_tokens, {[], state.context, state.current_token, nil}, fn index,
                                                                                       {steps,
                                                                                        context,
                                                                                        current_token,
                                                                                        sampler_state} ->
        {step_time, step} =
          timed("step_#{index}", fn ->
            Llamex.step(context, current_token, %{
              sampler: sampler,
              sampler_state: sampler_state,
              history: context.tokens
            })
          end)

        step_info = %{
          index: index,
          token: step.token,
          piece: Map.fetch!(model.tokenizer.id_to_token, step.token),
          text: step.text,
          timing: step_time
        }

        {[step_info | steps], step.context, step.token, step.sampler_state}
      end)

    steps = Enum.reverse(steps)
    generated_tokens = Enum.map(steps, & &1.token)

    %{
      prompt_tokens: length(state.prompt_tokens),
      generated_tokens: generated_tokens,
      text: Llamex.decode(model, generated_tokens),
      timings: [prefill_time | Enum.map(steps, & &1.timing)],
      steps: steps
    }
  end
end
