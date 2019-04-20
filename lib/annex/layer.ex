defmodule Annex.Layer do
  @type backprop_output :: {Data.t(), Keyword.t(), struct()}
  @type t() :: struct()

  @callback feedforward(struct(), Data.t()) :: {Data.t(), struct()}
  @callback backprop(struct(), float(), Data.t(), Keyword.t()) :: backprop_output()
  @callback init_layer(struct(), Keyword.t()) :: {:ok, struct()} | {:error, any()}
  @callback encoder() :: module()

  # @spec feedforward(struct(), Data.t()) :: {Data.t(), struct()}
  def feedforward(%module{} = layer, inputs) do
    inputs = encoder(layer).encode(inputs)
    {_, _} = module.feedforward(layer, inputs)
  end

  @spec backprop(struct(), float(), Data.t(), Keyword.t()) :: backprop_output()
  def backprop(%module{} = layer, total_loss_pd, loss_pds, layer_opts) do
    loss_pds = encoder(layer).encode(loss_pds)
    {_, _, _} = module.backprop(layer, total_loss_pd, loss_pds, layer_opts)
  end

  def init(%module{} = layer, opts \\ []) do
    module.init_layer(layer, opts)
  end

  def encoder(%module{}) do
    module.encoder()
  end

  def encoder(%module{}) do
    module.encoder()
  end
end
