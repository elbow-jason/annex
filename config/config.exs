use Mix.Config

config :annex,
  defaults: [
    learning_rate: 0.05,
    cost: Annex.Cost.MeanSquaredError
  ]

config :annex, Annex.Layer.Dense, data_type: Annex.Data.DMatrix

import_config("#{Mix.env()}.exs")
