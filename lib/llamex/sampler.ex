defmodule Llamex.Sampler do
  @moduledoc """
  Token sampling strategies.
  """

  def greedy(logits, backend) when is_atom(backend) do
    backend.argmax(logits)
  end

  def sample(logits, backend, opts) when is_atom(backend) and is_map(opts) do
    values = backend.to_list(logits)
    temperature = Map.fetch!(opts, :temperature)
    random = Map.fetch!(opts, :random)

    if temperature <= 0.0 do
      raise ArgumentError, "temperature must be greater than zero"
    end

    values
    |> apply_temperature(temperature)
    |> apply_top_k(Map.get(opts, :top_k))
    |> probabilities()
    |> draw(random)
  end

  defp apply_temperature(values, temperature) do
    Enum.map(values, &(&1 / temperature))
  end

  defp apply_top_k(values, nil), do: values

  defp apply_top_k(values, top_k) when is_integer(top_k) and top_k > 0 do
    threshold =
      values
      |> Enum.sort(:desc)
      |> Enum.at(top_k - 1)

    Enum.map(values, fn value ->
      if value >= threshold do
        value
      else
        :negative_infinity
      end
    end)
  end

  defp probabilities(values) do
    finite_values = Enum.reject(values, &(&1 == :negative_infinity))
    max = Enum.max(finite_values)

    weighted =
      Enum.map(values, fn
        :negative_infinity -> 0.0
        value -> :math.exp(value - max)
      end)

    total = Enum.sum(weighted)

    Enum.map(weighted, &(&1 / total))
  end

  defp draw(probabilities, random) when is_float(random) and random >= 0.0 and random < 1.0 do
    probabilities
    |> Enum.with_index()
    |> Enum.reduce_while(random, fn {probability, index}, remaining ->
      if probability > 0.0 and remaining <= probability do
        {:halt, index}
      else
        {:cont, remaining - probability}
      end
    end)
  end
end
