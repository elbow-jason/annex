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
  @callback data_type(t()) :: Data.type() | nil
  @callback shape(t()) :: Shape.t() | nil

  @spec feedforward(struct(), any()) :: {struct(), any()}
  def feedforward(%module{} = layer, inputs) do
    module.feedforward(layer, inputs)
  end

  @spec backprop(struct(), any(), Backprop.t()) :: {struct(), any(), Backprop.t()}
  def backprop(%module{} = layer, error, props) do
    module.backprop(layer, error, props)
  end

  @spec init_layer(struct(), Keyword.t()) :: {:ok, struct()} | {:error, any()}
  def init_layer(%module{} = layer, opts) do
    module.init_layer(layer, opts)
  end

  @spec data_type(atom | struct()) :: Data.type()
  def data_type(%module{} = layer), do: module.data_type(layer)

  @spec shape(t()) :: Shape.t() | nil
  def shape(%module{} = layer), do: module.shape(layer)

  @spec convert(t(), Data.data(), Shape.t()) :: {:ok, Data.data()} | {:error, any()}
  def convert(layer, data, shape) do
    layer
    |> data_type()
    |> Data.convert(data, shape)
  end

  @spec forward_shape(t()) :: {pos_integer, :any} | nil
  def forward_shape(layer) do
    layer
    |> shape()
    |> case do
      nil -> nil
      shape -> {Shape.resolve_columns(shape), :any}
    end
  end

  @spec backward_shape(t()) :: {:any, pos_integer} | nil
  def backward_shape(layer) do
    layer
    |> shape()
    |> case do
      nil -> nil
      shape -> {:any, Shape.resolve_rows(shape)}
    end
  end
end
