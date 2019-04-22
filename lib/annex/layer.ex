defmodule Annex.Layer do
  alias Annex.{Backprop, Data}

  @type t() :: struct()

  @callback feedforward(struct(), Data.t()) :: {Data.t(), struct()}
  @callback backprop(struct(), Backdrop.t()) :: {struct(), Backprop.t()}
  @callback init_layer(struct(), Keyword.t()) :: {:ok, struct()} | {:error, any()}
  @callback encoder() :: module()

  @spec feedforward(struct(), Data.t()) :: {Data.t(), struct()}
  def feedforward(%module{} = layer, inputs) do
    inputs = encoder(layer).encode(inputs)
    {_, _} = module.feedforward(layer, inputs)
  end

  @spec backprop(struct(), Backprop.t()) :: {struct(), Backprop.t()}
  def backprop(%module{} = layer, %Backprop{} = backprops) do
    backprops = encode_loss_pds(layer, backprops)
    {_, _} = module.backprop(layer, backprops)
  end

  defp encode_loss_pds(layer, backprops) do
    layer_encoder = encoder(layer)

    loss_pds =
      backprops
      |> Backprop.get_loss_pds()
      |> layer_encoder.encode()

    Backprop.put_loss_pds(backprops, loss_pds)
  end

  @spec init(struct(), Keyword.t()) :: struct()
  def init(%module{} = layer, opts \\ []) do
    module.init_layer(layer, opts)
  end

  @spec encoder(struct()) :: module()
  def encoder(%module{}) do
    module.encoder()
  end
end
