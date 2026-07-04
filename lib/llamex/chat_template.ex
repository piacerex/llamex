defmodule Llamex.ChatTemplate do
  @moduledoc """
  Minimal chat prompt formatting for tokenizer templates.
  """

  @chatml_template "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

  def apply(nil, prompt) when is_binary(prompt), do: prompt

  def apply(@chatml_template, prompt) when is_binary(prompt) do
    "<|im_start|>user\n" <> prompt <> "<|im_end|>\n<|im_start|>assistant\n"
  end

  def apply(template, _prompt) when is_binary(template) do
    raise ArgumentError, "unsupported chat template"
  end
end
