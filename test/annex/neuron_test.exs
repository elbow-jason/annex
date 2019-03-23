defmodule Annex.NeuronTest do
  use ExUnit.Case
  alias Annex.{Neuron, Activation}

  test "new/2" do
    weights = [1.0, 1.0, 2.0]
    bias = 1.1

    assert Neuron.new(weights, bias) == %Neuron{
             weights: weights,
             bias: bias,
             sum: 0.0
           }
  end

  test "new_random/1" do
    assert %Neuron{
             bias: bias,
             weights: weights,
             sum: sum
           } = Neuron.new_random(4)

    assert is_float(sum)
    assert sum == 0.0
    assert length(weights) == 4
    assert Enum.all?(weights, &is_float/1)
    assert is_float(bias)
  end

  test "feedforward/2" do
    weights = [1.0, 0.0, -1.1]
    bias = 1.0
    n1 = Neuron.new(weights, bias)
    inputs = [1.0, 0.9, 0.0]
    n2 = Neuron.feedforward(n1, inputs)
    assert n1 == %Neuron{weights: weights}
    assert n2 == %Neuron{weights: weights, inputs: inputs, sum: 2.0}
  end

  test "backprop/5" do
    weights = [1.0, 0.0, -1.1]
    bias = 1.0
    n1 = Neuron.new(weights, bias)
    assert n1 == %Neuron{weights: weights, bias: bias}
    inputs = [1.0, 0.9, 0.0]
    n2 = Neuron.feedforward(n1, inputs)
    assert n2 == %Neuron{weights: weights, inputs: inputs, sum: 2.0}
    total_loss_pd = 0.5
    neuron_loss_pd = 0.3
    learn_rate = 0.05
    activation_deriv = &Activation.sigmoid/1

    assert {next_loss_pds, n3} =
             Neuron.backprop(n2, total_loss_pd, neuron_loss_pd, learn_rate, activation_deriv)

    assert n3 == %Neuron{
             inputs: inputs,
             bias: 0.9933940219151659,
             weights: [0.9933940219151659, -0.0059453802763507054, -1.1],
             sum: 2.0
           }

    assert next_loss_pds == [0.8807970779778823, 0.0, -0.9688767857756706]
  end
end
