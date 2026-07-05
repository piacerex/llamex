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

    {prefill_time, {state, prefill_timings}} = timed_prefill(model, prompt, backend)

    {step_time, step} =
      timed("step", fn ->
        Llamex.step(state.context, state.current_token, %{sampler: sampler})
      end)

    %{
      prompt_tokens: length(state.prompt_tokens),
      token: step.token,
      text: step.text,
      prefill_timings: prefill_timings,
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
    stop_tokens = stop_tokens(opts)

    {prefill_time, {state, prefill_timings}} = timed_prefill(model, prompt, backend)

    {steps, _context, _current_token, _sampler_state, finish_reason} =
      Enum.reduce_while(
        1..max_new_tokens,
        {[], state.context, state.current_token, nil, :length},
        fn index, {steps, context, current_token, sampler_state, _finish_reason} ->
          {step_time, step} =
            timed("step_#{index}", fn ->
              Llamex.step(context, current_token, %{
                sampler: sampler,
                sampler_state: sampler_state,
                history: context.tokens
              })
            end)

          step_info =
            model
            |> token_info(step.token)
            |> Map.merge(%{
              index: index,
              text: step.text,
              timing: step_time
            })

          finish_reason = if stop_token?(step.token, stop_tokens), do: :stop, else: :length

          next_state =
            {[step_info | steps], step.context, step.token, step.sampler_state, finish_reason}

          if stop_token?(step.token, stop_tokens) do
            {:halt, next_state}
          else
            {:cont, next_state}
          end
        end
      )

    steps = Enum.reverse(steps)
    generated_tokens = Enum.map(steps, & &1.token)

    %{
      backend: backend,
      max_new_tokens: max_new_tokens,
      stop_token: List.first(stop_tokens),
      stop_tokens: stop_tokens,
      sampler: sampler,
      prompt_tokens: length(state.prompt_tokens),
      prompt_token_ids: state.prompt_tokens,
      prompt_pieces: token_pieces(model, state.prompt_tokens),
      generated_tokens: generated_tokens,
      generated_pieces: token_pieces(model, generated_tokens),
      generated_token_info: Enum.map(generated_tokens, &token_info(model, &1)),
      finish_reason: finish_reason,
      text: Llamex.decode(model, generated_tokens),
      prefill_timings: prefill_timings,
      timings: [prefill_time | Enum.map(steps, & &1.timing)],
      steps: steps
    }
  end

  defp timed_prefill(model, prompt, backend) do
    timed("prefill", fn ->
      {encode_time, prompt_tokens} =
        timed("prompt_encode", fn ->
          Llamex.encode(model, prompt)
        end)

      {prepare_time, context} =
        timed("backend_prepare", fn ->
          Llamex.Context.new(model, backend)
        end)

      {prompt_eval_time, context} =
        timed("prompt_eval", fn ->
          prompt_tokens
          |> Enum.drop(-1)
          |> Enum.reduce(context, fn token, context ->
            {context, _logits} = Llamex.Engine.eval(context, token)
            context
          end)
        end)

      state = %{
        context: context,
        prompt_tokens: prompt_tokens,
        current_token: seed_token(prompt_tokens)
      }

      {state, [encode_time, prepare_time, prompt_eval_time]}
    end)
  end

  defp seed_token([]), do: raise(ArgumentError, "prompt must encode to at least one token")
  defp seed_token(prompt_tokens), do: List.last(prompt_tokens)

  defp token_pieces(model, token_ids) do
    Enum.map(token_ids, &Map.fetch!(model.tokenizer.id_to_token, &1))
  end

  defp stop_tokens(%{stop_tokens: stop_tokens}) when is_list(stop_tokens), do: stop_tokens
  defp stop_tokens(%{stop_token: nil}), do: []
  defp stop_tokens(%{stop_token: stop_token}) when is_integer(stop_token), do: [stop_token]
  defp stop_tokens(_opts), do: []

  defp stop_token?(token, stop_tokens), do: token in stop_tokens

  defp token_info(model, token_id) do
    model.tokenizer
    |> token_type(token_id)
    |> Map.merge(%{
      token: token_id,
      piece: Map.fetch!(model.tokenizer.id_to_token, token_id)
    })
  end

  defp token_type(tokenizer, token_id) do
    case Enum.find(tokenizer.token_types, &(&1.id == token_id)) do
      nil -> %{}
      %{type: type, type_id: type_id} -> %{type: type, type_id: type_id}
    end
  end
end
