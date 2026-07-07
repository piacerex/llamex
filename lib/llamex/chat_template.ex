defmodule Llamex.ChatTemplate do
  @moduledoc """
  Minimal chat prompt formatting for tokenizer templates.
  """

  import Kernel, except: [apply: 2, apply: 3]

  @chatml_template "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
  @supported_roles ["system", "user", "assistant"]

  def supported_roles, do: @supported_roles

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
    apply(@chatml_template, [%{role: "user", content: prompt}])
  end

  def apply(@chatml_template, messages) when is_list(messages) do
    validate_messages!(messages)

    Enum.map_join(messages, "", fn message ->
      "<|im_start|>" <>
        message_role(message) <> "\n" <> message_content(message) <> "<|im_end|>\n"
    end) <> "<|im_start|>assistant\n"
  end

  def apply(template, prompt) when is_binary(template) and is_binary(prompt) do
    raise ArgumentError, "unsupported chat template"
  end

  def apply(template, messages) when is_binary(template) and is_list(messages) do
    raise ArgumentError, "unsupported chat template"
  end

  def apply(template, prompt, tokenizer) when is_binary(template) and is_binary(prompt) do
    apply(template, [%{role: "user", content: prompt}], tokenizer)
  end

  def apply(template, messages, tokenizer) when is_binary(template) and is_list(messages) do
    validate_messages!(messages)

    cond do
      template == @chatml_template ->
        apply(template, messages)

      role_marker_template?(template) ->
        apply_role_marker_template(messages, tokenizer)

      true ->
        raise ArgumentError, "unsupported chat template"
    end
  end

  defp role_marker_template?(template) do
    String.contains?(template, "<|user|>") and
      String.contains?(template, "<|assistant|>") and
      String.contains?(template, "eos_token")
  end

  defp apply_role_marker_template(messages, tokenizer) do
    eos_token = eos_token(tokenizer)

    Enum.map_join(messages, "", fn message ->
      case message_role(message) do
        "user" ->
          "<|user|>\n" <> message_content(message) <> eos_token

        "assistant" ->
          "<|assistant|>\n" <> message_content(message) <> eos_token

        "system" ->
          "<|user|>\n" <> message_content(message) <> eos_token
      end
    end) <> "<|assistant|>"
  end

  defp validate_messages!(messages) do
    Enum.each(messages, fn message ->
      role = message_role(message)
      _content = message_content(message)

      if role not in @supported_roles do
        raise ArgumentError,
              "unsupported chat role: #{role}; supported roles: #{Enum.join(@supported_roles, ", ")}"
      end
    end)
  end

  defp message_role(%{role: role}) when is_atom(role), do: Atom.to_string(role)
  defp message_role(%{role: role}) when is_binary(role), do: role
  defp message_role(%{"role" => role}) when is_atom(role), do: Atom.to_string(role)
  defp message_role(%{"role" => role}) when is_binary(role), do: role

  defp message_content(%{content: content}) when is_binary(content), do: content
  defp message_content(%{"content" => content}) when is_binary(content), do: content

  defp eos_token(tokenizer) do
    get_in(tokenizer.special_tokens, [:eos, :token]) || "</s>"
  end
end
