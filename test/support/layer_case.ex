defmodule Annex.LayerCase do
  @moduledoc """
  Imports LayerHelpers and aliases LayerConfig.
  """
  use ExUnit.CaseTemplate

  using do
    quote do
      import Annex.LayerHelpers
      alias Annex.LayerConfig
    end
  end
end
