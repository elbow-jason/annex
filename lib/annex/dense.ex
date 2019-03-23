defmodule Annex.Dense do
  alias Annex.{Dense, Layer, Neuron, Utils}

  @behaviour Layer

  defstruct neurons: nil,
            rows: nil,
            cols: nil,
            activation_derivative: nil,
            learning_rate: nil

  defp get_neurons(%Dense{neurons: neurons}), do: neurons

  defp get_learning_rate(%Dense{learning_rate: nil}, opts) do
    Keyword.get(opts, :learning_rate, 0.05)
  end

  defp get_learning_rate(%Dense{learning_rate: rate}, _) when is_float(rate) do
    rate
  end

  defp get_activation_derivative(%Dense{activation_derivative: nil}, opts) do
    der = Keyword.get(opts, :activation_derivative)
    next_layer = Keyword.get(opts, :next_layer)

    case {der, next_layer} do
      {nil, %module{} = next} ->
        module.get_derivative(next)

      {nil, _} ->
        raise "Dense activation_derivative not found"

      {der, _} ->
        der
    end
  end

  defp get_activation_derivative(%Dense{activation_derivative: a}, _) when is_function(a, 1) do
    a
  end

  def initialize(%Dense{rows: rows, cols: cols} = layer, opts \\ []) do
    neurons =
      case get_neurons(layer) do
        nil ->
          Enum.map(1..rows, fn _ -> Neuron.new_random(cols) end)

        found when is_list(found) ->
          found
      end

    activation_derivative = get_activation_derivative(layer, opts)
    learning_rate = get_learning_rate(layer, opts)

    initialized = %Dense{
      layer
      | neurons: neurons,
        activation_derivative: activation_derivative,
        learning_rate: learning_rate
    }

    {:ok, initialized}
  end

  def feedforward(%Dense{} = layer, inputs) do
    {output, neurons} =
      layer
      |> get_neurons()
      |> Enum.map(fn neuron ->
        neuron = Neuron.feedforward(neuron, inputs)
        {Neuron.get_sum(neuron), neuron}
      end)
      |> Enum.unzip()

    {output, %Dense{layer | neurons: neurons}}
  end

  def backprop(%Dense{} = layer, total_loss_pd, loss_pds, opts) do
    learning_rate = get_learning_rate(layer, opts)
    activation_derivative = get_activation_derivative(layer, opts)

    {next_loss_pd, neurons} =
      layer
      |> get_neurons()
      |> Utils.zip(List.wrap(loss_pds))
      |> Enum.map(fn {neuron, loss_pd} ->
        Neuron.backprop(neuron, total_loss_pd, loss_pd, learning_rate, activation_derivative)
      end)
      |> Enum.unzip()

    {List.flatten(next_loss_pd), [], %Dense{layer | neurons: neurons}}
  end
end
