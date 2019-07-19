use Mix.Config

config :annex, Annex.Data.DTensor, debug: true

config :git_hooks,
  hooks: [
    pre_commit: [
      verbose: true,
      tasks: [
        "mix format --check-formatted --dry-run --check-equivalent"
      ]
    ],
    pre_push: [
      verbose: true,
      tasks: [
        "mix clean",
        "mix compile --warnings-as-errors",
        "mix credo --strict",
        "mix dialyzer --halt-exit-status"
      ]
    ]
  ]
