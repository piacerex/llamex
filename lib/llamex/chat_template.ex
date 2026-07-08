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

  def supported?(template) when is_binary(template) do
    role_marker_template?(template) or header_marker_template?(template)
  end

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
    apply(template, [%{role: "user", content: prompt}])
  end

  def apply(template, messages) when is_binary(template) and is_list(messages) do
    validate_messages!(messages)

    if header_marker_template?(template) do
      apply_header_marker_template(template, messages)
    else
      raise ArgumentError, "unsupported chat template"
    end
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
        apply_role_marker_template(template, messages, tokenizer)

      header_marker_template?(template) ->
        apply_header_marker_template(template, messages)

      true ->
        raise ArgumentError, "unsupported chat template"
    end
  end

  defp role_marker_template?(template) do
    String.contains?(template, "<|user|>") and
      String.contains?(template, "<|assistant|>") and
      String.contains?(template, "eos_token")
  end

  defp header_marker_template?(template) do
    String.contains?(template, "<|start_header_id|>") and
      String.contains?(template, "<|end_header_id|>") and
      String.contains?(template, "<|eot_id|>")
  end

  defp apply_role_marker_template(template, messages, tokenizer) do
    eos_token = eos_token(tokenizer)

    Enum.map_join(messages, "", fn message ->
      case message_role(message) do
        "user" ->
          "<|user|>\n" <> message_content(message) <> eos_token

        "assistant" ->
          "<|assistant|>\n" <> message_content(message) <> eos_token

        "system" ->
          system_marker(template, message_content(message), eos_token)
      end
    end) <> "<|assistant|>"
  end

  defp system_marker(template, content, eos_token) do
    if String.contains?(template, "<|system|>") do
      "<|system|>\n" <> content <> eos_token
    else
      "<|user|>\n" <> content <> eos_token
    end
  end

  defp apply_header_marker_template(template, messages) do
    prefix =
      if String.contains?(template, "<|begin_of_text|>"),
        do: "<|begin_of_text|>",
        else: ""

    prefix <>
      Enum.map_join(messages, "", fn message ->
        "<|start_header_id|>" <>
          message_role(message) <>
          "<|end_header_id|>\n\n" <> message_content(message) <> "<|eot_id|>"
      end) <> "<|start_header_id|>assistant<|end_header_id|>\n\n"
  end

  defp validate_messages!(messages) do
    Enum.each(Enum.with_index(messages), fn {message, index} ->
      unless is_map(message) do
        raise ArgumentError, "chat message #{index} must be a map"
      end

      role = message_role(message)
      content = message_content(message)

      if role not in @supported_roles do
        raise ArgumentError,
              "unsupported chat role: #{format_role(role)}; supported roles: #{Enum.join(@supported_roles, ", ")}"
      end

      unless is_binary(content) do
        raise ArgumentError, "chat message #{index} content must be a string"
      end
    end)
  end

  defp message_role(%{role: role}) when is_atom(role), do: Atom.to_string(role)
  defp message_role(%{role: role}) when is_binary(role), do: role
  defp message_role(%{"role" => role}) when is_atom(role), do: Atom.to_string(role)
  defp message_role(%{"role" => role}) when is_binary(role), do: role
  defp message_role(_message), do: nil

  defp message_content(%{content: content}) when is_binary(content), do: content
  defp message_content(%{"content" => content}) when is_binary(content), do: content
  defp message_content(_message), do: nil

  defp format_role(nil), do: "missing"
  defp format_role(role), do: role

  defp eos_token(tokenizer) do
    get_in(tokenizer.special_tokens, [:eos, :token]) || "</s>"
  end
end
