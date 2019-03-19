defmodule Annex.Dense do
  alias Annex.{Dense, Layer, Neuron}

  @behaviour Layer

  defstruct neurons: nil,
            inputs: nil,
            rows: nil,
            cols: nil

  def get_neurons(%Dense{neurons: neurons}), do: neurons
  def get_inputs(%Dense{inputs: inputs}), do: inputs

  def initialize(%Dense{neurons: nil, rows: rows, cols: cols} = layer) do
    neurons = Enum.map(1..rows, fn _ -> Neuron.new_random(cols) end)
    {:ok, %Dense{layer | neurons: neurons}}
  end

  def initialize(%Dense{} = layer) do
    {:ok, layer}
  end

  def feedforward(%Dense{} = layer, inputs) do
    {output, neurons} =
      layer
      |> get_neurons()
      |> Enum.map(fn neuron ->
        neuron = Neuron.update_sum(neuron, inputs)
        {Neuron.get_sum(neuron), neuron}
      end)
      |> Enum.unzip()

    {output, %Dense{layer | neurons: neurons, inputs: inputs}}
  end

  def backprop(%Dense{} = layer, total_loss_pd, loss_pds, opts) do
    learning_rate = Keyword.fetch!(opts, :learning_rate)
    activation_derivative = Keyword.fetch!(opts, :activation_derivative)

    {next_loss_pd, neurons} =
      [get_neurons(layer), List.wrap(loss_pds)]
      |> Enum.zip()
      |> Enum.map(fn {neuron, loss_pd} ->
        Neuron.backprop(neuron, total_loss_pd, loss_pd, learning_rate, activation_derivative)
      end)
      |> Enum.unzip()

    {List.flatten(next_loss_pd), [], %Dense{layer | neurons: neurons}}
  end
end
