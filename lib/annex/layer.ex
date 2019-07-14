defmodule Annex.Layer do
  @moduledoc """
  The Annex.Layer is the module that defines types, callbacks, and helper for Layers.

  By implementing the Layer behaviour a struct/model can be used along side other
  Layers to compose the layers of a deep neural network.
  """

  alias Annex.{
    Data,
    Data.Shape,
    Layer.Backprop
  }

  @type t() :: struct()

  @callback feedforward(t(), Data.data()) :: {struct(), Data.data()}
  @callback backprop(t(), Data.data(), Backprop.t()) :: {t(), Data.data(), Backprop.t()}
  @callback init_layer(t(), Keyword.t()) :: {:ok, t()} | {:error, any()}
  @callback data_type() :: Data.type()
  @callback shapes(t()) :: {Shape.t(), Shape.t()}

  @spec feedforward(struct(), any()) :: {struct(), any()}
  def feedforward(%module{} = layer, inputs) do
    module.feedforward(layer, inputs)
  end

  @spec backprop(struct(), any(), Backprop.t()) :: {struct(), any(), Backprop.t()}
  def backprop(%module{} = layer, error, props) do
    module.backprop(layer, error, props)
  end

  @spec init_layer(struct(), Keyword.t()) :: struct()
  def init_layer(%module{} = layer, opts \\ []) do
    module.init_layer(layer, opts)
  end

  @spec data_type(atom | struct()) :: Data.t()
  def data_type(%module{}), do: data_type(module)
  def data_type(module) when is_atom(module), do: module.data_type()

  def shapes(%module{} = layer), do: module.shapes(layer)

  @spec convert(t(), Data.data(), Shape.t()) :: Data.data()
  def convert(layer, data, shape) do
    layer
    |> data_type()
    |> Data.convert(data, shape)
  end

  # defp do_convert(layer_shape, data_type, data) do
  #   with(
  #     {:ok, data_shape} <- fetch_data_shape(data_type, data),
  #     {:fit?, false} <- {:fit?, Shape.match?(data_shape, layer_shape)}
  #   ) do
  #     data
  #   else
  #     _ ->
  #       Data.cast(data_type, data, layer_shape)
  #   end
  # end

  # defp fetch_data_shape(data_type, data) do
  #   if Data.is_type?(data_type, data) do
  #     {:ok, Data.shape(data_type, data)}
  #   else
  #     :error
  #   end
  # end
end
