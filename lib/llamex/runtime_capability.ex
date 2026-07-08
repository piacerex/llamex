defmodule Llamex.RuntimeCapability do
  @moduledoc """
  Runtime loadability guard for models that carry GGUF diagnostic metadata.
  """

  def validate!(model) do
    case Map.get(model, :runtime_capability) do
      nil ->
        :ok

      %{loadable?: true} ->
        :ok

      capability ->
        raise ArgumentError, error_message(model, capability)
    end
  end

  defp error_message(model, capability) do
    [
      "model runtime is not loadable",
      "architecture=#{Map.get(model, :architecture, "unknown")}",
      "runtime=#{Map.get(capability, :runtime_status, "unknown")}",
      "blocking_groups=#{format_atoms(Map.get(capability, :blocking_issue_groups, []))}",
      "runtime_blockers=#{format_list(Map.get(capability, :runtime_blockers, []))}"
    ]
    |> Enum.join("; ")
  end

  defp format_atoms([]), do: "none"
  defp format_atoms(values), do: values |> Enum.map(&Atom.to_string/1) |> Enum.join(", ")

  defp format_list([]), do: "none"
  defp format_list(values), do: Enum.join(values, "; ")
end
