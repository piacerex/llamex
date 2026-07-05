defmodule Llamex.Engine do
  @moduledoc """
  Minimal inference loop.
  """

  alias Llamex.Context
  alias Llamex.Layers.{Attention, Linear, RMSNorm, SwiGLU}
  alias Llamex.Tensor

  def eval(%Context{} = context, token) when is_integer(token) and token >= 0 do
    hidden = Map.fetch!(context.model.token_embeddings, token)
    position = length(context.tokens)
    {context, hidden} = run_layers(context, hidden, position)

    hidden =
      maybe_apply_output_norm(hidden, context.model.output_norm, context.model.config.epsilon)

    logits =
      if context.model.output do
        Linear.forward(hidden, Map.fetch!(context.model.output, :weight), context.backend)
      else
        embedding_logits(context, hidden)
      end
      |> context.backend.from_list()

    {Context.append(context, token), logits}
  end

  def next_token(%Context{} = context, token, sampler) when is_function(sampler, 2) do
    {context, logits} = eval(context, token)
    {context, sampler.(logits, context.backend)}
  end

  def greedy_next_token(%Context{} = context, token) when is_integer(token) and token >= 0 do
    hidden = Map.fetch!(context.model.token_embeddings, token)
    position = length(context.tokens)
    {context, hidden} = run_layers(context, hidden, position)

    hidden =
      maybe_apply_output_norm(hidden, context.model.output_norm, context.model.config.epsilon)

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
        RMSNorm.forward(hidden, Map.fetch!(layer, :attention_norm), context.model.config.epsilon)

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

      hidden = Tensor.add(hidden, attention)
      hidden = maybe_apply_mlp(hidden, layer, context.model.config.epsilon, context.backend)

      {%{context | kv_cache: kv_cache}, hidden}
    end)
  end

  defp maybe_apply_mlp(hidden, %{feed_forward_norm: feed_forward_norm} = layer, epsilon, backend) do
    feed_forward =
      hidden
      |> RMSNorm.forward(feed_forward_norm, epsilon)
      |> SwiGLU.forward(layer, backend)

    Tensor.add(hidden, feed_forward)
  end

  defp maybe_apply_mlp(hidden, _layer, _epsilon, _backend), do: hidden

  defp maybe_apply_output_norm(hidden, nil, _epsilon), do: hidden

  defp maybe_apply_output_norm(hidden, output_norm, epsilon) do
    RMSNorm.forward(hidden, output_norm, epsilon)
  end

  defp embedding_logits(context, hidden) do
    0..(context.model.config.vocab_size - 1)
    |> Enum.map(fn candidate ->
      candidate_embedding = Map.fetch!(context.model.token_embeddings, candidate)

      Tensor.dot(hidden, candidate_embedding)
    end)
  end

  defp greedy_token(%{backend: Llamex.Backend.List, model: %{output: %{weight: weight}}}, hidden) do
    Tensor.argmax_matvec(weight, hidden)
  end

  defp greedy_token(%{backend: backend, model: %{output: %{weight: weight}}}, hidden) do
    hidden
    |> Linear.forward(weight, backend)
    |> Llamex.Backend.List.argmax()
  end

  defp greedy_token(context, hidden) do
    0..(context.model.config.vocab_size - 1)
    |> Enum.reduce(nil, fn candidate, best ->
      candidate_embedding = Map.fetch!(context.model.token_embeddings, candidate)
      value = Tensor.dot(hidden, candidate_embedding)

      case best do
        nil -> {candidate, value}
        {_best_token, best_value} when value > best_value -> {candidate, value}
        best -> best
      end
    end)
    |> elem(0)
  end
end
