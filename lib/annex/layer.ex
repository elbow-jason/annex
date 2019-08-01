defmodule Annex.Layer do
  @moduledoc """
  The Annex.Layer is the module that defines types, callbacks, and helper for Layers.

  By implementing the Layer behaviour a struct/model can be used along side other
  Layers to compose the layers of a deep neural network.
  """

  alias Annex.{
    Data,
    Layer.Backprop,
    LayerConfig,
    Shape
  }

  @type t() :: struct()

  @callback feedforward(t(), Data.data()) :: {struct(), Data.data()}
  @callback backprop(t(), Data.data(), Backprop.t()) :: {t(), Data.data(), Backprop.t()}

  @callback init_layer(LayerConfig.t(module())) :: t()

  @callback data_type(t()) :: Data.type()
  @callback shapes(t()) :: {Shape.t(), Shape.t()}

  @optional_callbacks [
    shapes: 1,
    data_type: 1
  ]

  defmacro __using__(_) do
    quote do
      alias Annex.Layer
      @behaviour Layer

      alias Annex.AnnexError
      alias Annex.LayerConfig
      require Annex.Utils
      import Annex.Utils, only: [validate: 3]

      def __annex__(:is_layer?), do: true
    end
  end

  @spec feedforward(t(), Data.data()) :: {t(), Data.data()}
  def feedforward(%module{} = layer, inputs) do
    module.feedforward(layer, inputs)
  end

  @spec backprop(t(), Data.data(), Backprop.t()) :: {t(), Data.data(), Backprop.t()}
  def backprop(%module{} = layer, error, props) do
    module.backprop(layer, error, props)
  end

  @spec init_layer(LayerConfig.t(module())) :: t()
  def init_layer(%LayerConfig{} = cfg), do: LayerConfig.init_layer(cfg)

  @spec has_data_type?(module() | struct()) :: boolean()
  def has_data_type?(%module{}), do: has_data_type?(module)
  def has_data_type?(module) when is_atom(module), do: function_exported?(module, :data_type, 1)

  @spec data_type(atom | struct()) :: Data.type()
  def data_type(%module{} = layer), do: module.data_type(layer)

  @spec shapes(t()) :: {Shape.t(), Shape.t()}
  def shapes(%module{} = layer), do: module.shapes(layer)

  @spec has_shapes?(module() | struct()) :: boolean()
  def has_shapes?(%module{}), do: has_shapes?(module)
  def has_shapes?(module) when is_atom(module), do: function_exported?(module, :shapes, 1)

  @spec is_layer?(any()) :: boolean()
  def is_layer?(%module{}) do
    is_layer?(module)
  end

  def is_layer?(item) do
    is_atom(item) && function_exported?(item, :__annex__, 1) && item.__annex__(:is_layer?)
  end

  def input_shape(layer) do
    if has_shapes?(layer) do
      {input_shape, _} = shapes(layer)
      input_shape
    end
  end

  def output_shape(layer) do
    if has_shapes?(layer) do
      {_, output_shape} = shapes(layer)
      output_shape
    end
  end

  @spec convert(t(), Data.data(), Shape.t()) :: Data.data()
  def convert(layer, data, shape) do
    layer
    |> data_type()
    |> Data.convert(data, shape)
  end

  @spec forward_shape(t()) :: Shape.t() | nil
  def forward_shape(layer) do
    layer
    |> input_shape()
    |> case do
      nil -> nil
      shape -> [Shape.resolve_columns(shape), :any]
    end
  end

  @spec backward_shape(t()) :: Shape.t() | nil
  def backward_shape(layer) do
    layer
    |> input_shape()
    |> case do
      nil ->
        nil

      shape ->
        [:any, Shape.resolve_rows(shape)]
    end
  end
end
