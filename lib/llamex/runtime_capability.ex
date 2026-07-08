defmodule Llamex.RuntimeCapability do
  @moduledoc """
  Runtime loadability guard for models that carry GGUF diagnostic metadata.
  """

  def loadable?(model_or_capability)

  def loadable?(%{runtime_capability: capability}) when is_map(capability),
    do: loadable?(capability)

  def loadable?(%{loadable?: loadable?}), do: loadable?
  def loadable?(_model_or_capability), do: true

  def blocker_ids(model_or_capability) do
    model_or_capability
    |> runtime_blocker_details()
    |> Enum.map(& &1.id)
  end

  def blockers_by_component(model_or_capability) do
    model_or_capability
    |> runtime_blocker_details()
    |> Enum.group_by(& &1.component)
  end

  def feature_status(model_or_capability) do
    model_or_capability
    |> runtime_capability()
    |> Map.get(:runtime_feature_status, %{})
  end

  def blocked_features(model_or_capability) do
    model_or_capability
    |> feature_status()
    |> Enum.filter(fn {_feature, status} -> status == "blocked" end)
    |> Enum.map(fn {feature, _status} -> feature end)
    |> Enum.sort()
  end

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
      "runtime_blockers=#{format_list(Map.get(capability, :runtime_blockers, []))}",
      "blocked_features=#{format_atoms(blocked_features(capability))}",
      "runtime_blocker_details=#{format_blocker_details(Map.get(capability, :runtime_blocker_details, []))}"
    ]
    |> Enum.join("; ")
  end

  defp runtime_capability(%{runtime_capability: capability}) when is_map(capability),
    do: capability

  defp runtime_capability(capability) when is_map(capability), do: capability
  defp runtime_capability(_model_or_capability), do: %{}

  defp runtime_blocker_details(%{runtime_capability: capability}) when is_map(capability) do
    runtime_blocker_details(capability)
  end

  defp runtime_blocker_details(%{runtime_blocker_details: details}) when is_list(details) do
    details
  end

  defp runtime_blocker_details(_model_or_capability), do: []

  defp format_atoms([]), do: "none"
  defp format_atoms(values), do: values |> Enum.map(&Atom.to_string/1) |> Enum.join(", ")

  defp format_list([]), do: "none"
  defp format_list(values), do: Enum.join(values, "; ")

  defp format_blocker_details([]), do: "none"

  defp format_blocker_details(details) do
    details
    |> Enum.map(fn detail -> "#{detail.id}:#{detail.component}:#{detail.reason}" end)
    |> Enum.join("/")
  end
end
