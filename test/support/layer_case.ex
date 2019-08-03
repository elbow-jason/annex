defmodule Annex.LayerCase do
  use ExUnit.CaseTemplate

  using do
    quote do
      import Annex.LayerHelpers
      alias Annex.LayerConfig
    end
  end
end
