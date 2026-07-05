defmodule Llamex.MixProject do
  use Mix.Project

  def project do
    [
      app: :llamex,
      version: "0.1.0",
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.12.1", optional: true},
      {:exla, "~> 0.12.0", optional: true}
    ]
  end
end
