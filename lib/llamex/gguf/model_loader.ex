defmodule Llamex.GGUF.ModelLoader do
  @moduledoc """
  Loads Llamex models from GGUF files.

  F32, F16, BF16, and dequantized Q2_K/Q3_K/Q4_0/Q4_1/Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0/Q8_1/Q8_K tensor data are supported at this stage.
  """

  def load(path) when is_binary(path) do
    load(path, [])
  end

  def load(path, opts) when is_binary(path) and is_list(opts) do
    binary = File.read!(path)
    gguf = Llamex.GGUF.Reader.read_binary(binary)
    validate_loadable!(gguf)

    model_map = to_model_map(gguf, binary, opts)

    case Keyword.get(opts, :tensor_format, :dequantized) do
      :compact ->
        Llamex.ModelLoader.from_compact_map(model_map, opts)

      :dequantized ->
        Llamex.ModelLoader.from_map(model_map)

      tensor_format ->
        raise ArgumentError, "unsupported GGUF tensor format: #{inspect(tensor_format)}"
    end
  end

  def to_model_map(%Llamex.GGUF.Reader{} = gguf, binary) when is_binary(binary) do
    to_model_map(gguf, binary, [])
  end

  def to_model_map(%Llamex.GGUF.Reader{} = gguf, binary, opts)
      when is_binary(binary) and is_list(opts) do
    architecture = metadata_value(gguf.metadata, "general.architecture", nil)
    tensor_format = Keyword.get(opts, :tensor_format, :dequantized)
    tensors = tensors_from_reader(gguf, binary, architecture, tensor_format)

    %{
      "config" => config_from_metadata(gguf.metadata),
      "architecture" => architecture,
      "runtime_capability" => runtime_capability_summary(gguf),
      "tokenizer" => tokenizer_from_metadata(gguf.metadata),
      "tensor_schema" => tensor_schema_summary(gguf),
      "tensor_format" => Atom.to_string(tensor_format),
      "tensors" => tensors
    }
  end

  def tensor_schema_summary(%Llamex.GGUF.Reader{} = gguf) do
    architecture = metadata_value(gguf.metadata, "general.architecture", nil)
    tensor_names = Enum.map(gguf.tensors, & &1.name)

    Llamex.GGUF.TensorSchema.summary(architecture, tensor_names)
  end

  def tensor_schema_summary_file(path) when is_binary(path) do
    path
    |> Llamex.GGUF.Reader.read_metadata()
    |> tensor_schema_summary()
  end

  def model_config_summary(%Llamex.GGUF.Reader{} = gguf) do
    config_from_metadata(gguf.metadata)
  end

  def model_config_report(%Llamex.GGUF.Reader{} = gguf) do
    Llamex.GGUF.ModelConfig.report(gguf.metadata)
  end

  def runtime_capability_summary(%Llamex.GGUF.Reader{} = gguf) do
    gguf
    |> Llamex.GGUF.Diagnostic.inspect_reader()
    |> Map.fetch!(:runtime_capability)
  end

  def model_config_summary_file(path) when is_binary(path) do
    path
    |> Llamex.GGUF.Reader.read_metadata()
    |> model_config_summary()
  end

  def model_config_report_file(path) when is_binary(path) do
    path
    |> Llamex.GGUF.Reader.read_metadata()
    |> model_config_report()
  end

  def runtime_capability_summary_file(path) when is_binary(path) do
    path
    |> Llamex.GGUF.Reader.read_metadata()
    |> runtime_capability_summary()
  end

  defp tensors_from_reader(%Llamex.GGUF.Reader{} = gguf, binary, architecture, :dequantized) do
    gguf
    |> Llamex.GGUF.Reader.read_tensor_data(binary)
    |> then(&Llamex.GGUF.TensorSchema.normalize_tensor_map(architecture, &1))
  end

  defp tensors_from_reader(%Llamex.GGUF.Reader{} = gguf, binary, architecture, :compact) do
    gguf
    |> Llamex.GGUF.Reader.read_compact_tensor_data(binary)
    |> then(&Llamex.GGUF.TensorSchema.normalize_tensor_map(architecture, &1))
  end

  defp tensors_from_reader(_gguf, _binary, _architecture, tensor_format) do
    raise ArgumentError, "unsupported GGUF tensor format: #{inspect(tensor_format)}"
  end

  defp config_from_metadata(metadata) do
    Llamex.GGUF.ModelConfig.from_metadata(metadata)
  end

  defp tokenizer_from_metadata(metadata) do
    tokenizer = Llamex.GGUF.Tokenizer.from_metadata(metadata)

    %{
      "type" => tokenizer_type(tokenizer),
      "unknown_token" => tokenizer.unknown_token,
      "vocab" => tokenizer.token_to_id,
      "special_tokens" => tokenizer.special_tokens,
      "token_types" => tokenizer.token_types,
      "chat_template" => chat_template_from_metadata(metadata) || tokenizer.chat_template
    }
    |> put_merges(tokenizer)
  end

  defp chat_template_from_metadata(metadata) do
    metadata_value(metadata, "tokenizer.chat_template", nil) ||
      metadata_value(metadata, "tokenizer.ggml.chat_template", nil)
  end

  defp tokenizer_type(%Llamex.Tokenizer.BPE{}), do: "bpe"
  defp tokenizer_type(%Llamex.Tokenizer.Whitespace{}), do: "whitespace"

  defp put_merges(attrs, %Llamex.Tokenizer.BPE{merges: merges}) do
    Map.put(attrs, "merges", Enum.map(merges, fn {left, right} -> [left, right] end))
  end

  defp put_merges(attrs, _tokenizer), do: attrs

  defp validate_loadable!(gguf) do
    diagnostic = Llamex.GGUF.Diagnostic.inspect_reader(gguf)

    if diagnostic.loadable? do
      :ok
    else
      issues = Enum.join(diagnostic.compatibility_issues, "; ")

      blocking_groups =
        diagnostic.blocking_issue_groups
        |> Enum.map(&Atom.to_string/1)
        |> Enum.join(", ")

      raise ArgumentError,
            "GGUF model is not loadable by Llamex: #{issues} (blocking issue groups: #{blocking_groups})#{runtime_blockers_suffix(diagnostic)}#{blocked_runtime_features_suffix(diagnostic)}"
    end
  end

  defp runtime_blockers_suffix(%{architecture_runtime_blockers: []}), do: ""

  defp runtime_blockers_suffix(%{architecture_runtime_blockers: blockers}) do
    " (architecture runtime blockers: #{Enum.join(blockers, "; ")})"
  end

  defp blocked_runtime_features_suffix(%{runtime_capability: %{blocked_runtime_features: []}}),
    do: ""

  defp blocked_runtime_features_suffix(%{
         runtime_capability: %{blocked_runtime_features: features}
       }) do
    " (blocked runtime features: #{features |> Enum.map(&Atom.to_string/1) |> Enum.join(", ")})"
  end

  defp metadata_value(metadata, key, default) do
    case Map.fetch(metadata, key) do
      {:ok, %{value: value}} -> value
      :error -> default
    end
  end
end
