defmodule Llamex.ChatTemplate do
  @moduledoc """
  Minimal chat prompt formatting for tokenizer templates.
  """

  import Kernel, except: [apply: 2, apply: 3]

  @chatml_template "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

  def supported?(nil), do: true
  def supported?(@chatml_template), do: true
  def supported?(template) when is_binary(template), do: role_marker_template?(template)

  def markers(nil), do: []

  def markers(template) when is_binary(template) do
    ~r/<\|[^>]+\|>/
    |> Regex.scan(template)
    |> Enum.map(&List.first/1)
    |> Enum.uniq()
  end

  def missing_tokens(nil, _token_to_id), do: []

  def missing_tokens(template, token_to_id) when is_binary(template) and is_map(token_to_id) do
    template
    |> markers()
    |> Enum.reject(&Map.has_key?(token_to_id, &1))
  end

  def apply(nil, prompt) when is_binary(prompt), do: prompt

  def apply(@chatml_template, prompt) when is_binary(prompt) do
    "<|im_start|>user\n" <> prompt <> "<|im_end|>\n<|im_start|>assistant\n"
  end

  def apply(template, _prompt) when is_binary(template) do
    raise ArgumentError, "unsupported chat template"
  end

  def apply(template, prompt, tokenizer) when is_binary(template) and is_binary(prompt) do
    cond do
      template == @chatml_template ->
        apply(template, prompt)

      role_marker_template?(template) ->
        "<|user|>\n" <> prompt <> eos_token(tokenizer) <> "<|assistant|>"

      true ->
        raise ArgumentError, "unsupported chat template"
    end
  end

  defp role_marker_template?(template) do
    String.contains?(template, "<|user|>") and
      String.contains?(template, "<|assistant|>") and
      String.contains?(template, "eos_token")
  end

  defp eos_token(tokenizer) do
    get_in(tokenizer.special_tokens, [:eos, :token]) || "</s>"
  end
end
