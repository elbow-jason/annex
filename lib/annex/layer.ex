defmodule Annex.Layer do
  @type data :: Encoder.data()
  alias Annex.{Layer, Neuron, Activation, Utils}

  @type backprop_output :: {data(), Keyword.t(), struct()}

  @callback feedforward(struct(), data()) :: {data(), struct()}
  @callback backprop(struct(), float(), data, Keyword.t()) :: backprop_output()
  @callback initialize(struct(), Keyword.t()) :: {:ok, struct()} | {:error, any()}
  @callback encoder() :: module()

  # <<<<<<< HEAD
  # defstruct neurons: nil,
  #           inputs: nil

  # def get_neurons(%Layer{neurons: neurons}), do: neurons
  # def get_inputs(%Layer{inputs: inputs}), do: inputs

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
    # =======
    #   def feedforward(%module{} = layer, inputs) do
    #     inputs = encoder(layer).encode(inputs)
    #     module.feedforward(layer, inputs)
    #   end

    #   def backprop(%module{} = layer, total_loss_pd, loss_pds, layer_opts) do
    #     loss_pds = encoder(layer).encode(loss_pds)
    #     module.backprop(layer, total_loss_pd, loss_pds, layer_opts)
    #   end

    #   def initialize(%module{} = layer, opts \\ []) do
    #     module.initialize(layer, opts)
    # >>>>>>> a9d0c6c... added iris
  end

  def encoder(%module{}) do
    module.encoder()
  end
end
