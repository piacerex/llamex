defmodule Llamex.GGUF.ModelLoader do
  @moduledoc """
  Loads Llamex models from GGUF files.

  F32, F16, BF16, and dequantized Q2_K/Q3_K/Q4_0/Q4_1/Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0/Q8_1/Q8_K tensor data are supported at this stage.
  """

  def load(path) when is_binary(path) do
    binary = File.read!(path)
    gguf = Llamex.GGUF.Reader.read_binary(binary)
    validate_loadable!(gguf)

    gguf
    |> to_model_map(binary)
    |> Llamex.ModelLoader.from_map()
  end

  def to_model_map(%Llamex.GGUF.Reader{} = gguf, binary) when is_binary(binary) do
    architecture = metadata_value(gguf.metadata, "general.architecture", nil)
    tensors = tensors_from_reader(gguf, binary, architecture)

    %{
      "config" => config_from_metadata(gguf.metadata),
      "tokenizer" => tokenizer_from_metadata(gguf.metadata),
      "tensor_schema" => tensor_schema_summary(gguf),
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
    %{
      "metadata_prefix" => Llamex.GGUF.ModelConfig.metadata_prefix(gguf.metadata),
      "config" => model_config_summary(gguf)
    }
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

  defp tensors_from_reader(%Llamex.GGUF.Reader{} = gguf, binary, architecture) do
    gguf
    |> Llamex.GGUF.Reader.read_tensor_data(binary)
    |> then(&Llamex.GGUF.TensorSchema.normalize_tensor_map(architecture, &1))
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
      "chat_template" => tokenizer.chat_template
    }
    |> put_merges(tokenizer)
  end

  defp tokenizer_type(%Llamex.Tokenizer.BPE{}), do: "bpe"
  defp tokenizer_type(%Llamex.Tokenizer.Whitespace{}), do: "whitespace"

  defp put_merges(attrs, %Llamex.Tokenizer.BPE{merges: merges}) do
    Map.put(attrs, "merges", Enum.map(merges, fn {left, right} -> [left, right] end))
  end

  defp put_merges(attrs, _tokenizer), do: attrs

  defp validate_loadable!(gguf) do
    if Llamex.GGUF.Diagnostic.loadable?(gguf) do
      :ok
    else
      issues =
        gguf
        |> Llamex.GGUF.Diagnostic.compatibility_issues()
        |> Enum.join("; ")

      raise ArgumentError, "GGUF model is not loadable by Llamex: #{issues}"
    end
  end

  defp metadata_value(metadata, key, default) do
    case Map.fetch(metadata, key) do
      {:ok, %{value: value}} -> value
      :error -> default
    end
  end
end
