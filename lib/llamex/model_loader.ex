defmodule Llamex.ModelLoader do
  @moduledoc """
  Loads toy Llamex models from files.
  """

  def load_json(path) when is_binary(path) do
    path
    |> File.read!()
    |> JSON.decode!()
    |> atomize_model()
    |> Llamex.new_model()
  end

  defp atomize_model(attrs) when is_map(attrs) do
    %{
      config: atomize_keys(Map.fetch!(attrs, "config")),
      token_embeddings: Map.fetch!(attrs, "token_embeddings")
    }
    |> put_optional(attrs, "layers", :layers)
    |> put_optional(attrs, "output", :output)
    |> put_tokenizer(attrs)
    |> integer_key_embeddings()
  end

  defp put_tokenizer(attrs, %{"tokenizer" => tokenizer}) when is_map(tokenizer) do
    tokenizer =
      Llamex.Tokenizer.new(Map.fetch!(tokenizer, "vocab"), Map.fetch!(tokenizer, "unknown_token"))

    Map.put(attrs, :tokenizer, tokenizer)
  end

  defp put_tokenizer(attrs, _source), do: attrs

  defp put_optional(attrs, source, source_key, target_key) do
    if Map.has_key?(source, source_key) do
      Map.put(attrs, target_key, atomize_value(Map.fetch!(source, source_key)))
    else
      attrs
    end
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
      {String.to_atom(key), atomize_value(value)}
    end)
  end

  defp atomize_value(value) when is_map(value), do: atomize_keys(value)
  defp atomize_value(values) when is_list(values), do: Enum.map(values, &atomize_value/1)
  defp atomize_value(value), do: value

  defp parse_token_id(token) when is_integer(token), do: token

  defp parse_token_id(token) when is_binary(token) do
    case Integer.parse(token) do
      {id, ""} -> id
      _other -> raise ArgumentError, "token embedding keys must be integer strings"
    end
  end
end
