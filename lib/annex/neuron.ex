defmodule Annex.Neuron do
  alias Annex.{Neuron, Utils}

  defstruct weights: [],
            bias: 1.0,
            sum: 0.0,
            output: 0.0,
            inputs: nil

  def new(weights, bias) do
    %Neuron{
      weights: weights,
      bias: bias
    }
  end

  def new_random(size) when is_integer(size) and size > 0 do
    weights = Utils.random_weights(size)
    bias = Utils.random_float()
    new(weights, bias)
  end

  def get_bias(%Neuron{bias: bias}), do: bias
  def get_weights(%Neuron{weights: w}), do: w
  def get_output(%Neuron{output: o}), do: o
  def get_sum(%Neuron{sum: sum}), do: sum
  def get_inputs(%Neuron{inputs: inputs}), do: inputs

  def feedforward(%Neuron{} = neuron, inputs) do
    bias = get_bias(neuron)

    sum =
      neuron
      |> get_weights
      |> Enum.zip(inputs)
      |> Enum.map(fn {w, i} -> w * i end)
      |> Enum.sum()
      |> Kernel.+(bias)

    %Neuron{neuron | sum: sum, inputs: inputs}
  end

  def backprop(%Neuron{} = neuron, total_loss_pd, neuron_loss_pd, learning_rate, activation_deriv) do
    sum = get_sum(neuron)
    bias = get_bias(neuron)
    sum_deriv = activation_deriv.(sum)
    inputs = get_inputs(neuron)
    weights = get_weights(neuron)
    delta_coeff = learning_rate * total_loss_pd * neuron_loss_pd

    {[_ | next_neuron_loss], [new_bias | new_weights]} =
      [1.0 | inputs]
      |> Utils.zip([bias | weights])
      |> Enum.map(fn {input, weight} ->
        delta = input * sum_deriv * delta_coeff
        {weight * sum_deriv, weight - delta}
      end)
      |> Enum.unzip()

    {next_neuron_loss, %Neuron{neuron | weights: new_weights, bias: new_bias}}
  end
end
