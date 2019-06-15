defmodule Annex.MixProject do
  use Mix.Project

  def project do
    [
      app: :annex,
      version: "0.1.0",
      elixir: "~> 1.8",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      elixirc_paths: elixirc_paths(Mix.env()),
      description: description(),
      package: package(),
      deps: deps(),
      name: "Annex",
      source_url: "https://github.com/elbow-jason/annex"
    ]
  end

  def elixirc_paths(:dev), do: ["lib", "examples"]
  def elixirc_paths(_), do: ["lib"]

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nimble_csv, "~> 0.3", only: [:dev]},
      {:ex_doc, "~> 0.14", only: :dev, runtime: false}
    ]
  end

  defp description() do
    "A composable deep learning framework in Elixir"
  end

  defp package() do
    [
      # This option is only needed when you don't want to use the OTP application name
      name: "annex",
      # These are the default files included in the package
      files: ~w(lib .formatter.exs mix.exs README.md LICENSE CHANGELOG.md),
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/elbow-jason/annex"}
    ]
  end
end
