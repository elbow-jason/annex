defmodule Annex.Layer do
  @type data :: [float(), ...] | reference()

  @type backprop_output :: {data(), Keyword.t(), struct()}

  @callback feedforward(struct(), data()) :: {data(), struct()}
  @callback backprop(struct(), float(), data, Keyword.t()) :: backprop_output()
  @callback initialize(struct(), Keyword.t()) :: {:ok, struct()} | {:error, any()}

  alias Annex.{Layer}

  defstruct neurons: nil,
            inputs: nil

  def get_neurons(%Layer{neurons: neurons}), do: neurons
  def get_inputs(%Layer{inputs: inputs}), do: inputs

  def feedforward(%module{} = layer, inputs) do
    module.feedforward(layer, inputs)
  end

  def backprop(%module{} = layer, total_loss_pd, loss_pds, layer_opts) do
    module.backprop(layer, total_loss_pd, loss_pds, layer_opts)
  end

  def initialize(%module{} = layer, opts \\ []) do
    module.initialize(layer, opts)
  end
end
