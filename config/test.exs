use Mix.Config

config :annex, Annex.Data.DTensor, debug: true

config :annex, Annex.Layer.Dense, data_type: Annex.Data.DMatrix
