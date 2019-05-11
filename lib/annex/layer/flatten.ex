defmodule Annex.Layer.Flatten do
  alias Annex.{Utils, Layer, Layer.Flatten, Layer.Backprop}
  @behaviour Layer

  defstruct prev_shape: []

  def encoder, do: Annex.Data

  def init_layer(%Flatten{} = layer, _opts) do
    {:ok, layer}
  end

  def feedforward(%Flatten{} = layer, inputs) do
    {List.flatten(inputs), layer}
  end

  def backprop(%Flatten{prev_shape: shape} = layer, %Backprop{} = bp) do
    loss_pds = Backprop.get_loss_pds(bp)
    reshaped = reshape_loss_pds(shape, loss_pds)
    {layer, Backprop.put_loss_pds(bp, reshaped)}
  end

  defp reshape_loss_pds(shape, loss_pds) do
    shape
    |> Enum.reverse()
    |> Utils.reshape(loss_pds)
  end
end
