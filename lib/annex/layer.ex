defmodule Annex.Layer do
  @type data :: Encoder.data()

  @type backprop_output :: {data(), Keyword.t(), struct()}

  @callback feedforward(struct(), data()) :: {data(), struct()}
  @callback backprop(struct(), float(), data, Keyword.t()) :: backprop_output()
  @callback initialize(struct(), Keyword.t()) :: {:ok, struct()} | {:error, any()}
  @callback encoder() :: module()

  def feedforward(%module{} = layer, inputs) do
    inputs = encoder(layer).encode(inputs)
    module.feedforward(layer, inputs)
  end

  def backprop(%module{} = layer, total_loss_pd, loss_pds, layer_opts) do
    loss_pds = encoder(layer).encode(loss_pds)
    module.backprop(layer, total_loss_pd, loss_pds, layer_opts)
  end

  def initialize(%module{} = layer, opts \\ []) do
    module.initialize(layer, opts)
  end

  def encoder(%module{}) do
    module.encoder()
  end

  def encoder(%module{}) do
    module.encoder()
  end
end
