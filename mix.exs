defmodule Annex.MixProject do
  use Mix.Project

  def project do
    [
      app: :annex,
      version: "0.2.0",
      elixir: "~> 1.8",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      elixirc_paths: elixirc_paths(Mix.env()),
      aliases: aliases(),
      # package
      description: description(),
      package: package(),
      name: "Annex",
      source_url: "https://github.com/elbow-jason/annex",
      # coverage
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: [
        coveralls: :test,
        "coveralls.detail": :test,
        "coveralls.post": :test,
        "coveralls.html": :test
      ],
      dialyzer: [
        ignore_warnings: ".dialyzer_ignore.exs",
        list_unused_filters: true,
        plt_file: {:no_warn, "annex.plt"}
      ]
    ]
  end

  def elixirc_paths(:dev), do: ["lib", "examples"]
  def elixirc_paths(:test), do: ["lib", "examples", "test/support"]
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
      {:git_hooks, "~> 0.3.1", only: :dev, runtime: false},
      {:excoveralls, "~> 0.11.1", only: :test},
      {:credo, "~> 1.1.0", only: :dev, runtime: false},
      {:mix_test_watch, "~> 0.9.0", only: :dev, runtime: false},
      {:dialyxir, "~> 1.0.0-rc.6", only: :dev, runtime: false},
      {:earmark, "~> 1.3.2", only: :dev, runtime: false},
      {:nimble_csv, "~> 0.6.0", only: [:dev, :test]},
      {:ex_doc, "~> 0.20.2", only: :dev, runtime: false},
      {:tensor, "~> 2.1"},
      {:map_array, "~> 0.1.0"},
      {:mox, "~> 0.5.0"}
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

  defp aliases do
    [
      "git.checks": ["git_hooks.run all"],
      coverage: [&coveralls_html_and_open_browser/1]
    ]
  end

  defp coveralls_html_and_open_browser(_) do
    Mix.Task.run("coveralls.html", [])

    System.cmd(open_command(), ["cover/excoveralls.html"])
  end

  defp open_command do
    case :os.type() do
      {:win32, _} ->
        "start"

      {:unix, :darwin} ->
        "open"

      {:unix, _} ->
        "xdg-open"
    end
  end
end
