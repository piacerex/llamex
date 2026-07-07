defmodule Llamex.StopSequences do
  @moduledoc false

  def from_options(%{stop_sequences: stop_sequences}) when is_list(stop_sequences) do
    Enum.flat_map(stop_sequences, &normalize_sequence/1)
  end

  def from_options(%{stop_sequences: stop_sequences}) do
    raise ArgumentError,
          "stop_sequences must be a list of strings, got: #{inspect(stop_sequences)}"
  end

  def from_options(%{stop_sequence: stop_sequence}) do
    normalize_sequence(stop_sequence)
  end

  def from_options(_opts), do: []

  defp normalize_sequence(sequence) when is_binary(sequence) do
    if sequence == "", do: [], else: [sequence]
  end

  defp normalize_sequence(sequence) do
    raise ArgumentError, "stop sequence must be a string, got: #{inspect(sequence)}"
  end
end
