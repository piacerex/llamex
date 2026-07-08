defmodule Llamex.ChatTemplate do
  @moduledoc """
  Minimal chat prompt formatting for tokenizer templates.
  """

  import Kernel, except: [apply: 2, apply: 3]

  @chatml_template "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
  @supported_roles ["system", "user", "assistant"]
  @supported_families ["chatml", "role_markers", "llama_header_markers", "gemma_turn_markers"]

  def supported_roles, do: @supported_roles

  def supported_families, do: @supported_families

  def supported?(nil), do: true
  def supported?(@chatml_template), do: true

  def supported?(template) when is_binary(template) do
    role_marker_template?(template) or header_marker_template?(template) or
      gemma_turn_marker_template?(template)
  end

  def family(nil), do: "none"

  def family(template) when is_binary(template) do
    cond do
      not supported?(template) ->
        "unsupported"

      chatml_template?(template) ->
        "chatml"

      header_marker_template?(template) ->
        "llama_header_markers"

      gemma_turn_marker_template?(template) ->
        "gemma_turn_markers"

      role_marker_template?(template) ->
        "role_markers"

      true ->
        "supported"
    end
  end

  def markers(nil), do: []

  def markers(template) when is_binary(template) do
    pipe_markers =
      ~r/<\|[^>]+\|>/
      |> Regex.scan(template)
      |> Enum.map(&List.first/1)

    gemma_markers =
      ~r/<(?:start|end)_of_turn>/
      |> Regex.scan(template)
      |> Enum.map(&List.first/1)

    Enum.uniq(pipe_markers ++ gemma_markers)
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

      gemma_turn_marker_template?(template) ->
        apply_gemma_turn_marker_template(messages)

      true ->
        raise ArgumentError, "unsupported chat template"
    end
  end

  defp role_marker_template?(template) do
    String.contains?(template, "<|user|>") and
      String.contains?(template, "<|assistant|>") and
      String.contains?(template, "eos_token")
  end

  defp chatml_template?(template) do
    String.contains?(template, "<|im_start|>") and
      String.contains?(template, "<|im_end|>")
  end

  defp header_marker_template?(template) do
    String.contains?(template, "<|start_header_id|>") and
      String.contains?(template, "<|end_header_id|>") and
      String.contains?(template, "<|eot_id|>")
  end

  defp gemma_turn_marker_template?(template) do
    String.contains?(template, "<start_of_turn>") and
      String.contains?(template, "<end_of_turn>") and
      String.contains?(template, "model")
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

  defp apply_gemma_turn_marker_template(messages) do
    prompt =
      messages
      |> merge_system_into_gemma_user()
      |> Enum.map_join("", fn message ->
        case message_role(message) do
          "assistant" ->
            "<start_of_turn>model\n" <> message_content(message) <> "<end_of_turn>\n"

          role ->
            "<start_of_turn>" <> role <> "\n" <> message_content(message) <> "<end_of_turn>\n"
        end
      end)

    prompt <> "<start_of_turn>model\n"
  end

  defp merge_system_into_gemma_user(messages) do
    {system_messages, turn_messages} =
      Enum.split_while(messages, &(message_role(&1) == "system"))

    system_content =
      system_messages
      |> Enum.map(&message_content/1)
      |> Enum.reject(&(&1 in [nil, ""]))
      |> Enum.join("\n\n")

    case {system_content, turn_messages} do
      {"", messages} ->
        messages

      {system_content, [first | rest]} ->
        if message_role(first) == "user" do
          [Map.put(first, :content, system_content <> "\n\n" <> message_content(first)) | rest]
        else
          [%{role: "user", content: system_content}, first | rest]
        end

      {system_content, messages} ->
        [%{role: "user", content: system_content} | messages]
    end
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
