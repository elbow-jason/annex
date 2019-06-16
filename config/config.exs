use Mix.Config

config :annex,
  defaults: [
    learning_rate: 0.05,
    cost: Annex.Cost.MeanSquaredError
  ]

import_config("#{Mix.env()}.exs")
