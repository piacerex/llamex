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

  def from_compact_map(%{"tensor_format" => "compact", "tensors" => tensors} = attrs)
      when is_map(tensors) do
    from_compact_map(attrs, [])
  end

  def from_compact_map(%{"tensor_format" => tensor_format}) do
    raise ArgumentError, "compact model map expected tensor_format=compact, got #{tensor_format}"
  end

  def from_compact_map(_attrs) do
    raise ArgumentError, "compact model map expected tensor_format=compact"
  end

  def from_compact_map(%{"tensor_format" => "compact", "tensors" => tensors} = attrs, opts)
      when is_map(tensors) and is_list(opts) do
    compact_backend? = Keyword.get(opts, :compact_backend, false)

    attrs
    |> Map.put("token_embeddings", Llamex.TensorStore.fetch_dequantized_token_embeddings(tensors))
    |> maybe_put_compact_layers(tensors, compact_backend?)
    |> maybe_put_compact_output(tensors, compact_backend?)
    |> Map.delete("tensors")
    |> from_map()
  end

  defp maybe_put_compact_output(%{"output" => _output} = attrs, _tensors, _compact_backend?) do
    attrs
  end

  defp maybe_put_compact_output(attrs, tensors, compact_backend?) do
    if Map.has_key?(tensors, "output.weight") do
      Map.put(attrs, "output", %{
        "weight" => compact_matrix_value(tensors, "output.weight", compact_backend?)
      })
    else
      attrs
    end
  end

  defp maybe_put_compact_layers(%{"layers" => _layers} = attrs, _tensors, _compact_backend?) do
    attrs
  end

  defp maybe_put_compact_layers(attrs, tensors, compact_backend?) do
    layers =
      tensors
      |> Llamex.TensorStore.layer_count()
      |> layer_indexes()
      |> Enum.map(&compact_layer_from_tensors(tensors, attrs, &1, compact_backend?))

    if layers == [] do
      attrs
    else
      Map.put(attrs, "layers", layers)
    end
  end

  defp compact_layer_from_tensors(tensors, attrs, index, compact_backend?) do
    wq = compact_matrix_value(tensors, "blk.#{index}.attn_q.weight", compact_backend?)
    wk = compact_matrix_value(tensors, "blk.#{index}.attn_k.weight", compact_backend?)
    head_count = get_in(attrs, ["config", "attention_head_count"])

    %{
      "head_count" => head_count,
      "kv_head_count" => kv_head_count(attrs, head_count, wq, wk),
      "attention_norm" =>
        Llamex.TensorStore.fetch_dequantized_matrix(tensors, "blk.#{index}.attn_norm.weight"),
      "feed_forward_norm" =>
        fetch_optional_dequantized_matrix(tensors, "blk.#{index}.ffn_norm.weight"),
      "attention_q_norm" =>
        fetch_optional_dequantized_matrix(tensors, "blk.#{index}.attn_q_norm.weight"),
      "attention_k_norm" =>
        fetch_optional_dequantized_matrix(tensors, "blk.#{index}.attn_k_norm.weight"),
      "post_feed_forward_norm" =>
        fetch_optional_dequantized_matrix(tensors, "blk.#{index}.post_ffw_norm.weight"),
      "wq" => wq,
      "wk" => wk,
      "wv" => compact_matrix_value(tensors, "blk.#{index}.attn_v.weight", compact_backend?),
      "wo" => compact_matrix_value(tensors, "blk.#{index}.attn_output.weight", compact_backend?),
      "w_gate" =>
        fetch_optional_compact_matrix_value(
          tensors,
          "blk.#{index}.ffn_gate.weight",
          compact_backend?
        ),
      "w_up" =>
        fetch_optional_compact_matrix_value(
          tensors,
          "blk.#{index}.ffn_up.weight",
          compact_backend?
        ),
      "w_down" =>
        fetch_optional_compact_matrix_value(
          tensors,
          "blk.#{index}.ffn_down.weight",
          compact_backend?
        )
    }
    |> Enum.reject(fn {_key, value} -> is_nil(value) end)
    |> Map.new()
  end

  defp compact_matrix_value(tensors, name, true) do
    Llamex.TensorStore.fetch_compact_tensor(tensors, name)
  end

  defp compact_matrix_value(tensors, name, false) do
    Llamex.TensorStore.fetch_dequantized_matrix(tensors, name)
  end

  defp fetch_optional_compact_matrix_value(tensors, name, compact_backend?) do
    if Map.has_key?(tensors, name) do
      compact_matrix_value(tensors, name, compact_backend?)
    end
  end

  defp fetch_optional_dequantized_matrix(tensors, name) do
    if Map.has_key?(tensors, name) do
      Llamex.TensorStore.fetch_dequantized_matrix(tensors, name)
    end
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
    |> put_model_metadata(attrs)
    |> integer_key_embeddings()
  end

  defp put_model_metadata(attrs, source) do
    source
    |> Map.take(["architecture", "runtime_capability", "tensor_schema"])
    |> atomize_keys()
    |> then(&Map.merge(attrs, &1))
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
          token_types: atomize_token_types(Map.get(tokenizer, "token_types", [])),
          chat_template: Map.get(tokenizer, "chat_template")
        )

      "bpe" ->
        Llamex.Tokenizer.bpe(
          Map.fetch!(tokenizer, "vocab"),
          Map.fetch!(tokenizer, "merges"),
          Map.fetch!(tokenizer, "unknown_token"),
          special_tokens: atomize_special_tokens(Map.get(tokenizer, "special_tokens", %{})),
          token_types: atomize_token_types(Map.get(tokenizer, "token_types", [])),
          chat_template: Map.get(tokenizer, "chat_template")
        )

      type ->
        raise ArgumentError, "unsupported tokenizer type: #{type}"
    end
  end

  defp decoded_tensors(%{"config" => config, "tensors" => tensors}) do
    tensors
    |> Llamex.TensorStore.decode()
    |> Map.put(:__config__, atomize_keys(config))
  end

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
    wq = Llamex.TensorStore.fetch_matrix(tensors, "blk.#{index}.attn_q.weight")
    wk = Llamex.TensorStore.fetch_matrix(tensors, "blk.#{index}.attn_k.weight")
    head_count = tensor_config(tensors, :attention_head_count)

    %{
      head_count: head_count,
      kv_head_count: kv_head_count(tensors, head_count, wq, wk),
      attention_norm: Llamex.TensorStore.fetch_matrix(tensors, "blk.#{index}.attn_norm.weight"),
      feed_forward_norm:
        Llamex.TensorStore.fetch_optional_matrix(tensors, "blk.#{index}.ffn_norm.weight"),
      attention_q_norm:
        Llamex.TensorStore.fetch_optional_matrix(tensors, "blk.#{index}.attn_q_norm.weight"),
      attention_k_norm:
        Llamex.TensorStore.fetch_optional_matrix(tensors, "blk.#{index}.attn_k_norm.weight"),
      post_feed_forward_norm:
        Llamex.TensorStore.fetch_optional_matrix(tensors, "blk.#{index}.post_ffw_norm.weight"),
      wq: wq,
      wk: wk,
      wv: Llamex.TensorStore.fetch_matrix(tensors, "blk.#{index}.attn_v.weight"),
      wo: Llamex.TensorStore.fetch_matrix(tensors, "blk.#{index}.attn_output.weight"),
      w_gate: Llamex.TensorStore.fetch_optional_matrix(tensors, "blk.#{index}.ffn_gate.weight"),
      w_up: Llamex.TensorStore.fetch_optional_matrix(tensors, "blk.#{index}.ffn_up.weight"),
      w_down: Llamex.TensorStore.fetch_optional_matrix(tensors, "blk.#{index}.ffn_down.weight")
    }
    |> Enum.reject(fn {_key, value} -> is_nil(value) end)
    |> Map.new()
  end

  defp tensor_config(tensors, key) do
    tensors
    |> Map.get(:__config__, %{})
    |> Map.get(key)
  end

  defp kv_head_count(_tensors, nil, _wq, _wk), do: nil

  defp kv_head_count(tensors, head_count, wq, wk) do
    configured = tensor_config(tensors, :attention_head_count_kv)
    head_size = div(matrix_row_count(wq), head_count)

    cond do
      is_nil(configured) -> head_count
      matrix_row_count(wk) == configured * head_size -> configured
      true -> head_count
    end
  end

  defp matrix_row_count(%{info: %{shape: [rows | _rest]}}), do: rows
  defp matrix_row_count(rows) when is_list(rows), do: length(rows)

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

  defp atomize_token_types(token_types) when is_list(token_types) do
    Enum.map(token_types, fn token_type ->
      token_type
      |> atomize_value()
      |> normalize_token_type()
    end)
  end

  defp normalize_token_type(%{type: type} = token_type) when is_binary(type) do
    %{token_type | type: atomize_key(type)}
  end

  defp normalize_token_type(token_type), do: token_type

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
