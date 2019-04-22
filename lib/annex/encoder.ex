defmodule Annex.Encoder do
  alias Annex.Data

  @callback encode(Data.data()) :: Data.data()
  @callback decode(Data.data()) :: Data.float_data()
end
