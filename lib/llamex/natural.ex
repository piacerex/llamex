defmodule Llamex.Natural do
  @moduledoc """
  Natural text generation defaults shared by CLI tasks.
  """

  def sampler(model, opts \\ %{}) when is_map(opts) do
    opts
    |> Map.put_new(:temperature, 0.8)
    |> Map.put_new(:top_k, 40)
    |> Map.put_new(:top_p, 0.5)
    |> Map.put_new(:repetition_penalty, 1.1)
    |> Map.put_new(:seed, 1)
    |> put_suppressed_tokens(model)
  end

  def control_stop_tokens(%{tokenizer: nil}), do: []
  def control_stop_tokens(model) when not is_map_key(model, :tokenizer), do: []

  def control_stop_tokens(model) do
    model.tokenizer.token_types
    |> Enum.filter(&(&1.type == :control))
    |> Enum.map(& &1.id)
  end

  def smoke_check(model, generated_tokens, text, opts \\ %{})
      when is_list(generated_tokens) and is_binary(text) do
    min_words = Map.get(opts, :min_words, 1)
    has_text_content? = text_content?(text)

    issues =
      []
      |> add_issue(String.contains?(text, "▁"), "raw sentencepiece marker in text")
      |> add_issue(not has_text_content?, "no alphanumeric text generated")
      |> add_issue(
        has_text_content? and word_count(text) < min_words,
        "generated fewer than #{min_words} word(s)"
      )
      |> Kernel.++(token_issues(model, generated_tokens))

    %{ok: issues == [], issues: issues}
  end

  defp put_suppressed_tokens(sampler, model) do
    suppress_tokens = natural_suppressed_token_ids(model)

    if suppress_tokens == [] do
      sampler
    else
      Map.update(sampler, :suppress_tokens, suppress_tokens, &Enum.uniq(&1 ++ suppress_tokens))
    end
  end

  defp natural_suppressed_token_ids(%{tokenizer: nil}), do: []
  defp natural_suppressed_token_ids(model) when not is_map_key(model, :tokenizer), do: []

  defp natural_suppressed_token_ids(model) do
    type_ids =
      model.tokenizer.token_types
      |> Enum.filter(&natural_suppressed_token?/1)
      |> Enum.map(& &1.id)

    special_ids =
      [:unknown]
      |> Enum.map(&get_in(model.tokenizer.special_tokens, [&1, :id]))
      |> Enum.reject(&is_nil/1)

    control_ids =
      model.tokenizer.token_types
      |> Enum.filter(&(&1.type == :control))
      |> Enum.map(& &1.id)
      |> Enum.reject(&(&1 == get_in(model.tokenizer.special_tokens, [:eos, :id])))

    Enum.uniq(type_ids ++ special_ids ++ control_ids)
  end

  defp natural_suppressed_token?(%{type: type})
       when type in [:unknown, :unused, :user_defined, :undefined],
       do: true

  defp natural_suppressed_token?(%{type: :byte}), do: true
  defp natural_suppressed_token?(%{token: "▁"}), do: true

  defp natural_suppressed_token?(%{token: token}) when is_binary(token) do
    String.contains?(token, ["\n", "\r"])
  end

  defp natural_suppressed_token?(_token), do: false

  defp add_issue(issues, true, issue), do: [issue | issues]
  defp add_issue(issues, false, _issue), do: issues

  defp text_content?(text), do: Regex.match?(~r/[[:alnum:]]/u, text)

  defp word_count(text) do
    ~r/[[:alnum:]]+/u
    |> Regex.scan(text)
    |> length()
  end

  defp token_issues(%{tokenizer: nil}, _generated_tokens), do: []
  defp token_issues(model, _generated_tokens) when not is_map_key(model, :tokenizer), do: []

  defp token_issues(model, generated_tokens) do
    suppressed = MapSet.new(natural_suppressed_token_ids(model))
    token_types = Map.new(model.tokenizer.token_types, &{&1.id, &1})

    generated_tokens
    |> Enum.flat_map(fn token ->
      cond do
        MapSet.member?(suppressed, token) ->
          [suppressed_token_issue(token, token_types)]

        true ->
          []
      end
    end)
  end

  defp suppressed_token_issue(token, token_types) do
    case Map.fetch(token_types, token) do
      {:ok, %{type: type, token: piece}} ->
        "suppressed token generated: #{token}:#{piece}:#{type}"

      :error ->
        "suppressed token generated: #{token}"
    end
  end
end
