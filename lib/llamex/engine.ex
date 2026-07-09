defmodule Llamex.Engine do
  @moduledoc """
  Minimal inference loop.
  """

  alias Llamex.Context
  alias Llamex.Layers.{Attention, SwiGLU}
  alias Llamex.Tensor

  def eval(%Context{} = context, token) when is_integer(token) and token >= 0 do
    hidden = Map.fetch!(context.model.token_embeddings, token)
    position = context.token_count
    {context, hidden} = run_layers(context, hidden, position)

    hidden =
      maybe_apply_output_norm(
        hidden,
        context.model.output_norm,
        context.model.config.epsilon,
        context.backend
      )

    logits =
      if context.model.output do
        context.backend.matvec_tensor(Map.fetch!(context.model.output, :weight), hidden)
      else
        embedding_logits(context, hidden)
        |> context.backend.from_list()
      end

    {Context.append(context, token), logits}
  end

  def eval_top_k(
        %Context{model: %{output: %{weight: weight}}} = context,
        token,
        top_k,
        opts
      )
      when is_integer(token) and token >= 0 and is_integer(top_k) and top_k > 0 and is_map(opts) do
    hidden = Map.fetch!(context.model.token_embeddings, token)
    position = context.token_count
    {context, hidden} = run_layers(context, hidden, position)

    hidden =
      maybe_apply_output_norm(
        hidden,
        context.model.output_norm,
        context.model.config.epsilon,
        context.backend
      )

    candidates =
      context.backend.top_k_matvec(weight, hidden, top_k,
        history: Map.get(opts, :history, []),
        repetition_penalty: Map.get(opts, :repetition_penalty),
        suppress_tokens: Map.get(opts, :suppress_tokens, [])
      )

    {Context.append(context, token), candidates}
  end

  def next_token(%Context{} = context, token, sampler) when is_function(sampler, 2) do
    {context, logits} = eval(context, token)
    {context, sampler.(logits, context.backend)}
  end

  def greedy_next_token(%Context{} = context, token) when is_integer(token) and token >= 0 do
    hidden = Map.fetch!(context.model.token_embeddings, token)
    position = context.token_count
    {context, hidden} = run_layers(context, hidden, position)

    hidden =
      maybe_apply_output_norm(
        hidden,
        context.model.output_norm,
        context.model.config.epsilon,
        context.backend
      )

    next_token = greedy_token(context, hidden)

    {Context.append(context, token), next_token}
  end

  defp run_layers(%Context{model: %{layers: []}} = context, hidden, _position),
    do: {context, hidden}

  defp run_layers(%Context{} = context, hidden, position) do
    context.model.layers
    |> Enum.with_index()
    |> Enum.reduce({context, hidden}, fn {layer, layer_index}, {context, hidden} ->
      normalized =
        context.backend.rms_norm(
          hidden,
          Map.fetch!(layer, :attention_norm),
          context.model.config.epsilon
        )

      {kv_cache, attention} =
        Attention.forward(
          normalized,
          layer,
          context.kv_cache,
          layer_index,
          position,
          context.model.config.rope_theta,
          context.model.config.rope_dimension_count,
          context.backend
        )

      hidden = context.backend.add(hidden, attention)
      hidden = maybe_apply_mlp(hidden, layer, context.model.config.epsilon, context.backend)

      {%{context | kv_cache: kv_cache}, hidden}
    end)
  end

  defp maybe_apply_mlp(hidden, %{feed_forward_norm: feed_forward_norm} = layer, epsilon, backend) do
    feed_forward =
      hidden
      |> backend.rms_norm(feed_forward_norm, epsilon)
      |> SwiGLU.forward(layer, backend)
      |> maybe_apply_post_feed_forward_norm(layer, epsilon, backend)

    backend.add(hidden, feed_forward)
  end

  defp maybe_apply_mlp(hidden, _layer, _epsilon, _backend), do: hidden

  defp maybe_apply_post_feed_forward_norm(
         feed_forward,
         %{post_feed_forward_norm: norm},
         epsilon,
         backend
       ) do
    backend.rms_norm(feed_forward, norm, epsilon)
  end

  defp maybe_apply_post_feed_forward_norm(feed_forward, _layer, _epsilon, _backend),
    do: feed_forward

  defp maybe_apply_output_norm(hidden, nil, _epsilon, _backend), do: hidden

  defp maybe_apply_output_norm(hidden, output_norm, epsilon, backend) do
    backend.rms_norm(hidden, output_norm, epsilon)
  end

  defp embedding_logits(context, hidden) do
    0..(context.model.config.vocab_size - 1)
    |> Enum.map(fn candidate ->
      candidate_embedding = Map.fetch!(context.model.token_embeddings, candidate)

      context.backend.dot(hidden, candidate_embedding)
    end)
  end

  defp greedy_token(%{backend: Llamex.Backend.List, model: %{output: %{weight: weight}}}, hidden)
       when is_list(weight) do
    Tensor.argmax_matvec(weight, hidden)
  end

  defp greedy_token(%{backend: backend, model: %{output: %{weight: weight}}}, hidden) do
    weight
    |> backend.matvec_tensor(hidden)
    |> backend.argmax()
  end

  defp greedy_token(context, hidden) do
    0..(context.model.config.vocab_size - 1)
    |> Enum.reduce(nil, fn candidate, best ->
      candidate_embedding = Map.fetch!(context.model.token_embeddings, candidate)
      value = context.backend.dot(hidden, candidate_embedding)

      case best do
        nil -> {candidate, value}
        {_best_token, best_value} when value > best_value -> {candidate, value}
        best -> best
      end
    end)
    |> elem(0)
  end
end
