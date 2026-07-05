defmodule Llamex.Sampler do
  @moduledoc """
  Token sampling strategies.
  """

  def greedy(logits, backend) when is_atom(backend) do
    backend.argmax(logits)
  end

  def sample(logits, backend, opts) when is_atom(backend) and is_map(opts) do
    random = Map.fetch!(opts, :random)

    logits
    |> distribution(backend, opts)
    |> draw(random)
  end

  def sample_candidates(candidates, opts) when is_list(candidates) and is_map(opts) do
    random = Map.fetch!(opts, :random)

    candidates
    |> candidate_distribution(opts)
    |> draw(random)
  end

  def candidates(logits, backend, opts, limit)
      when is_atom(backend) and is_map(opts) and is_integer(limit) and limit > 0 do
    logits
    |> distribution(backend, opts)
    |> Enum.sort_by(fn {probability, _index} -> probability end, :desc)
    |> Enum.take(limit)
    |> Enum.map(fn {probability, index} -> %{token: index, probability: probability} end)
  end

  def candidate_probabilities(candidates, opts, limit)
      when is_list(candidates) and is_map(opts) and is_integer(limit) and limit > 0 do
    candidates
    |> candidate_distribution(opts)
    |> Enum.sort_by(fn {probability, _index} -> probability end, :desc)
    |> Enum.take(limit)
    |> Enum.map(fn {probability, index} -> %{token: index, probability: probability} end)
  end

  defp distribution(logits, backend, opts) do
    values = backend.to_list(logits)
    temperature = Map.fetch!(opts, :temperature)

    if temperature <= 0.0 do
      raise ArgumentError, "temperature must be greater than zero"
    end

    values
    |> apply_repetition_penalty(Map.get(opts, :history, []), Map.get(opts, :repetition_penalty))
    |> apply_temperature(temperature)
    |> top_k_candidates(Map.get(opts, :top_k))
    |> probabilities()
    |> apply_top_p(Map.get(opts, :top_p))
    |> normalize()
  end

  defp candidate_distribution(candidates, opts) do
    temperature = Map.fetch!(opts, :temperature)

    if temperature <= 0.0 do
      raise ArgumentError, "temperature must be greater than zero"
    end

    candidates
    |> Enum.map(fn {value, index} -> {value / temperature, index} end)
    |> probabilities()
    |> apply_top_p(Map.get(opts, :top_p))
    |> normalize()
  end

  defp apply_repetition_penalty(values, _history, nil), do: values

  defp apply_repetition_penalty(values, history, penalty)
       when is_list(history) and is_number(penalty) and penalty > 0.0 do
    repeated = MapSet.new(history)

    values
    |> Enum.with_index()
    |> Enum.map(fn {value, index} ->
      if MapSet.member?(repeated, index) do
        penalize(value, penalty)
      else
        value
      end
    end)
  end

  defp penalize(value, penalty) when value >= 0.0, do: value / penalty
  defp penalize(value, penalty), do: value * penalty

  defp apply_temperature(values, temperature) do
    Enum.map(values, &(&1 / temperature))
  end

  defp top_k_candidates(values, nil) do
    Enum.with_index(values)
  end

  defp top_k_candidates(values, top_k) when is_integer(top_k) and top_k > 0 do
    values
    |> Enum.with_index()
    |> top_k_by_value(top_k)
  end

  defp top_k_by_value(candidates, top_k) do
    candidates
    |> Enum.reduce([], fn candidate, top ->
      insert_top_k(candidate, top, top_k)
    end)
    |> Enum.reverse()
  end

  defp insert_top_k(candidate, [], _top_k), do: [candidate]

  defp insert_top_k({value, _index}, [{lowest, _lowest_index} | _rest] = top, top_k)
       when length(top) == top_k and value <= lowest do
    top
  end

  defp insert_top_k(candidate, top, top_k) do
    [candidate | top]
    |> Enum.sort_by(fn {value, _index} -> value end)
    |> trim_lowest(top_k)
  end

  defp trim_lowest(top, top_k) when length(top) > top_k, do: tl(top)
  defp trim_lowest(top, _top_k), do: top

  defp probabilities(candidates) do
    max = candidates |> Enum.map(fn {value, _index} -> value end) |> Enum.max()

    weighted =
      Enum.map(candidates, fn {value, index} ->
        {:math.exp(value - max), index}
      end)

    total = weighted |> Enum.map(fn {weight, _index} -> weight end) |> Enum.sum()

    Enum.map(weighted, fn {weight, index} -> {weight / total, index} end)
  end

  defp apply_top_p(probabilities, nil), do: probabilities

  defp apply_top_p(probabilities, top_p) when is_number(top_p) and top_p > 0.0 and top_p <= 1.0 do
    probabilities
    |> Enum.sort_by(fn {probability, _index} -> probability end, :desc)
    |> keep_until_top_p(top_p, 0.0, [])
  end

  defp keep_until_top_p([], _top_p, _total, kept), do: Enum.reverse(kept)

  defp keep_until_top_p([{probability, index} | rest], top_p, total, kept) do
    kept = [{probability, index} | kept]
    total = total + probability

    if total >= top_p do
      Enum.reverse(kept)
    else
      keep_until_top_p(rest, top_p, total, kept)
    end
  end

  defp normalize(probabilities) do
    total = probabilities |> Enum.map(fn {probability, _index} -> probability end) |> Enum.sum()

    if total == 0.0 do
      raise ArgumentError, "sampling filters removed all probabilities"
    end

    Enum.map(probabilities, fn {probability, index} -> {probability / total, index} end)
  end

  defp draw(probabilities, random) when is_float(random) and random >= 0.0 and random < 1.0 do
    probabilities
    |> Enum.reduce_while(random, fn {probability, index}, remaining ->
      if probability > 0.0 and remaining <= probability do
        {:halt, index}
      else
        {:cont, remaining - probability}
      end
    end)
  end
end
