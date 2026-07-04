defmodule Llamex.ModelLoader do
  @moduledoc """
  Loads toy Llamex models from files.
  """

  def load_json(path) when is_binary(path) do
    path
    |> File.read!()
    |> JSON.decode!()
    |> from_map()
  end

  def from_map(attrs) when is_map(attrs) do
    attrs
    |> atomize_model()
    |> Llamex.new_model()
  end

  defp atomize_model(attrs) when is_map(attrs) do
    tensors = decoded_tensors(attrs)

    %{
      config: atomize_keys(Map.fetch!(attrs, "config")),
      token_embeddings: token_embeddings(attrs, tensors)
    }
    |> put_layers(attrs, tensors)
    |> put_output_norm(attrs, tensors)
    |> put_output(attrs, tensors)
    |> put_tokenizer(attrs)
    |> integer_key_embeddings()
  end

  defp put_tokenizer(attrs, %{"tokenizer" => tokenizer}) when is_map(tokenizer) do
    tokenizer =
      case tokenizer do
        %{"path" => path} ->
          Llamex.Tokenizer.Loader.load_tokenizer_json(path)

        tokenizer ->
          tokenizer_from_attrs(tokenizer)
      end

    Map.put(attrs, :tokenizer, tokenizer)
  end

  defp put_tokenizer(attrs, _source), do: attrs

  defp tokenizer_from_attrs(tokenizer) do
    case Map.get(tokenizer, "type", "whitespace") do
      "whitespace" ->
        Llamex.Tokenizer.whitespace(
          Map.fetch!(tokenizer, "vocab"),
          Map.fetch!(tokenizer, "unknown_token"),
          special_tokens: atomize_special_tokens(Map.get(tokenizer, "special_tokens", %{})),
          token_types: atomize_value(Map.get(tokenizer, "token_types", []))
        )

      "bpe" ->
        Llamex.Tokenizer.bpe(
          Map.fetch!(tokenizer, "vocab"),
          Map.fetch!(tokenizer, "merges"),
          Map.fetch!(tokenizer, "unknown_token"),
          special_tokens: atomize_special_tokens(Map.get(tokenizer, "special_tokens", %{})),
          token_types: atomize_value(Map.get(tokenizer, "token_types", []))
        )

      type ->
        raise ArgumentError, "unsupported tokenizer type: #{type}"
    end
  end

  defp decoded_tensors(%{"tensors" => tensors}), do: Llamex.TensorStore.decode(tensors)
  defp decoded_tensors(_attrs), do: nil

  defp put_layers(attrs, %{"layers" => layers}, _tensors) do
    Map.put(attrs, :layers, atomize_value(layers))
  end

  defp put_layers(attrs, _source, nil), do: attrs

  defp put_layers(attrs, _source, tensors) do
    layers =
      tensors
      |> Llamex.TensorStore.layer_count()
      |> layer_indexes()
      |> Enum.map(&layer_from_tensors(tensors, &1))

    Map.put(attrs, :layers, layers)
  end

  defp layer_indexes(0), do: []
  defp layer_indexes(count), do: 0..(count - 1)

  defp put_output_norm(attrs, %{"output_norm" => %{"weight" => weight}}, _tensors) do
    Map.put(attrs, :output_norm, weight)
  end

  defp put_output_norm(attrs, %{"output_norm" => weight}, _tensors) when is_list(weight) do
    Map.put(attrs, :output_norm, weight)
  end

  defp put_output_norm(attrs, _source, nil), do: attrs

  defp put_output_norm(attrs, _source, tensors) do
    case Llamex.TensorStore.fetch_optional_matrix(tensors, "output_norm.weight") do
      nil -> attrs
      weight -> Map.put(attrs, :output_norm, weight)
    end
  end

  defp put_output(attrs, %{"output" => output}, _tensors) do
    Map.put(attrs, :output, atomize_value(output))
  end

  defp put_output(attrs, _source, nil), do: attrs

  defp put_output(attrs, _source, tensors) do
    case Llamex.TensorStore.fetch_optional_matrix(tensors, "output.weight") do
      nil -> attrs
      weight -> Map.put(attrs, :output, %{weight: weight})
    end
  end

  defp layer_from_tensors(tensors, index) do
    %{
      attention_norm: Llamex.TensorStore.fetch_matrix(tensors, "blk.#{index}.attn_norm.weight"),
      feed_forward_norm:
        Llamex.TensorStore.fetch_optional_matrix(tensors, "blk.#{index}.ffn_norm.weight"),
      wq: Llamex.TensorStore.fetch_matrix(tensors, "blk.#{index}.attn_q.weight"),
      wk: Llamex.TensorStore.fetch_matrix(tensors, "blk.#{index}.attn_k.weight"),
      wv: Llamex.TensorStore.fetch_matrix(tensors, "blk.#{index}.attn_v.weight"),
      wo: Llamex.TensorStore.fetch_matrix(tensors, "blk.#{index}.attn_output.weight"),
      w_gate: Llamex.TensorStore.fetch_optional_matrix(tensors, "blk.#{index}.ffn_gate.weight"),
      w_up: Llamex.TensorStore.fetch_optional_matrix(tensors, "blk.#{index}.ffn_up.weight"),
      w_down: Llamex.TensorStore.fetch_optional_matrix(tensors, "blk.#{index}.ffn_down.weight")
    }
    |> Enum.reject(fn {_key, value} -> is_nil(value) end)
    |> Map.new()
  end

  defp token_embeddings(%{"token_embeddings" => token_embeddings}, _tensors), do: token_embeddings

  defp token_embeddings(_attrs, tensors) when is_map(tensors) do
    tensors
    |> Llamex.TensorStore.fetch_matrix("token_embd.weight")
    |> Enum.with_index()
    |> Map.new(fn {embedding, token_id} -> {token_id, embedding} end)
  end

  defp integer_key_embeddings(%{token_embeddings: token_embeddings} = attrs) do
    token_embeddings =
      Map.new(token_embeddings, fn {token, embedding} ->
        {parse_token_id(token), embedding}
      end)

    %{attrs | token_embeddings: token_embeddings}
  end

  defp atomize_keys(map) when is_map(map) do
    Map.new(map, fn {key, value} ->
      {atomize_key(key), atomize_value(value)}
    end)
  end

  defp atomize_value(value) when is_map(value), do: atomize_keys(value)
  defp atomize_value(values) when is_list(values), do: Enum.map(values, &atomize_value/1)
  defp atomize_value(value), do: value

  defp atomize_special_tokens(tokens) when is_map(tokens) do
    Map.new(tokens, fn {key, value} ->
      {atomize_key(key), atomize_value(value)}
    end)
  end

  defp atomize_key(key) when is_atom(key), do: key
  defp atomize_key(key) when is_binary(key), do: String.to_atom(key)

  defp parse_token_id(token) when is_integer(token), do: token

  defp parse_token_id(token) when is_binary(token) do
    case Integer.parse(token) do
      {id, ""} -> id
      _other -> raise ArgumentError, "token embedding keys must be integer strings"
    end
  end
end
