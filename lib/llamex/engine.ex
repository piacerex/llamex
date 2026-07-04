defmodule Llamex.Engine do
  @moduledoc """
  Minimal inference loop.
  """

  alias Llamex.Context
  alias Llamex.Layers.{Attention, Linear, RMSNorm}
  alias Llamex.Tensor

  def eval(%Context{} = context, token) when is_integer(token) and token >= 0 do
    hidden = Map.fetch!(context.model.token_embeddings, token)
    {context, hidden} = run_layers(context, hidden)

    logits =
      if context.model.output do
        Linear.forward(hidden, Map.fetch!(context.model.output, :weight))
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

  defp run_layers(%Context{model: %{layers: []}} = context, hidden), do: {context, hidden}

  defp run_layers(%Context{} = context, hidden) do
    context.model.layers
    |> Enum.with_index()
    |> Enum.reduce({context, hidden}, fn {layer, layer_index}, {context, hidden} ->
      normalized =
        RMSNorm.forward(hidden, Map.fetch!(layer, :attention_norm), context.model.config.epsilon)

      {kv_cache, attention} = Attention.forward(normalized, layer, context.kv_cache, layer_index)
      hidden = Tensor.add(hidden, attention)

      {%{context | kv_cache: kv_cache}, hidden}
    end)
  end

  defp embedding_logits(context, hidden) do
    0..(context.model.config.vocab_size - 1)
    |> Enum.map(fn candidate ->
      candidate_embedding = Map.fetch!(context.model.token_embeddings, candidate)

      Tensor.dot(hidden, candidate_embedding)
    end)
  end
end
