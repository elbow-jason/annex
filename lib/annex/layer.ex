defmodule Annex.Layer do
  @moduledoc """
  The Annex.Layer is the module that defines types, callbacks, and helper for Layers.

  By implementing the Layer behaviour a struct/model can be used along side other
  Layers to compose the layers of a deep neural network.
  """

  alias Annex.{
    AnnexError,
    Data,
    Data.Shape,
    Layer.Backprop,
    LayerConfig
  }

  @type t() :: struct()

  @callback feedforward(t(), Data.data()) :: {struct(), Data.data()}
  @callback backprop(t(), Data.data(), Backprop.t()) :: {t(), Data.data(), Backprop.t()}

  @callback init_layer(LayerConfig.t(module())) :: {:ok, struct()} | {:error, AnnexError.t()}
  @callback data_type(t()) :: Data.type() | nil
  @callback shapes(t()) :: {Shape.t(), Shape.t()}

  @optional_callbacks [
    shapes: 1
  ]

  defmacro __using__(_) do
    quote do
      alias Annex.Layer
      @behaviour Layer

      alias Annex.AnnexError
      alias Annex.LayerConfig
      require Annex.Utils
      import Annex.Utils, only: [validate: 3]
    end
  end

  @spec feedforward(struct(), any()) :: {struct(), any()}
  def feedforward(%module{} = layer, inputs) do
    module.feedforward(layer, inputs)
  end

  @spec backprop(struct(), any(), Backprop.t()) :: {struct(), any(), Backprop.t()}
  def backprop(%module{} = layer, error, props) do
    module.backprop(layer, error, props)
  end

  @spec init_layer(struct()) :: {:ok, struct()} | {:error, AnnexError.t()}
  def init_layer(%LayerConfig{} = cfg) do
    LayerConfig.init_layer(cfg)
  end

  @spec data_type(atom | struct()) :: Data.type()
  def data_type(%module{} = layer), do: module.data_type(layer)

  @spec shape(t()) :: Shape.t() | nil
  def shapes(%module{} = layer), do: module.shape(layer)

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
