defmodule Annex.LayerHelpers do
  alias Annex.{
    AnnexError,
    LayerConfig
  }

  @type kvs :: map | keyword()

  @spec build(atom, kvs) :: struct()
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
