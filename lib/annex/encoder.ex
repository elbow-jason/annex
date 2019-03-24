defmodule Annex.Encoder do
  @type data :: [float(), ...] | struct()
  @callback encode(data()) :: data()
  @callback decode(data()) :: [float(), ...]
end
