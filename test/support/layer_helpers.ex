defmodule Annex.LayerHelpers do
  @moduledoc """
  Helpers for building Layers.
  """
  alias Annex.LayerConfig

  @type kvs :: map | keyword()

  @spec build(atom, kvs) :: Layer.t()
  def build(module, kvs) do
    module
    |> LayerConfig.build(kvs)
    |> LayerConfig.init_layer()
  end
end
