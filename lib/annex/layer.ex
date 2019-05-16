defmodule Annex.Layer do
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
  def backprop(%module{} = layer, loss_pds, props) do
    module.backprop(layer, loss_pds, props)
  end

  @spec init(struct(), Keyword.t()) :: struct()
  def init(%module{} = layer, opts \\ []) do
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
