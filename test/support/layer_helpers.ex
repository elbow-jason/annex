defmodule Annex.LayerHelpers do
  @moduledoc """
  Helpers for building Layers.
  """
  alias Annex.{
    AnnexError,
    LayerConfig
  }

  @type kvs :: map | keyword()

  @spec build(atom, kvs) :: {:ok, struct()} | {:error, AnnexError.t()}
  def build(module, kvs) do
    module
    |> LayerConfig.build(kvs)
    |> LayerConfig.init_layer()
  end

  @spec build!(atom, any) :: Layer.t()
  def build!(module, kvs) do
    module
    |> LayerConfig.build(kvs)
    |> LayerConfig.init_layer()
    |> case do
      {:ok, layer} ->
        layer

      {:error, %AnnexError{} = error} ->
        raise error
    end
  end
end
