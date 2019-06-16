defmodule Annex.Layer do
  @moduledoc """
  The Annex.Layer is the module that defines types, callbacks, and helper for Layers.

  By implementing the Layer behaviour a struct/model can be used along side other
  Layers to compose the layers of a deep neural network.
  """

  alias Annex.{
    Layer.Backprop
  }

  @type t() :: struct()
  @type data :: any()

  @callback feedforward(struct(), data()) :: {struct(), data()}
  @callback backprop(struct(), data(), Backprop.t()) :: {struct(), data(), Backprop.t()}
  @callback init_layer(struct(), Keyword.t()) :: {:ok, struct()} | {:error, any()}
  @callback encode(struct(), list(float)) :: any()
  @callback decode(struct(), any()) :: list(float)
  @callback encoded?(struct(), any) :: boolean()

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

  @spec convert(any(), struct(), struct()) :: any()
  def convert(data, %decoder{} = decoder_layer, %encoder{} = encoder_layer) do
    if encoder.encoded?(encoder_layer, data) do
      data
    else
      decoded = decoder.decode(decoder_layer, data)
      encoder.encode(encoder_layer, decoded)
    end
  end
end
