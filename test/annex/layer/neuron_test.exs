defmodule Annex.Layer.NeuronTest do
  use ExUnit.Case
  alias Annex.Layer.{Neuron, Activation}

  test "new/2" do
    weights = [1.0, 1.0, 2.0]
    bias = 1.1

    assert Neuron.new(weights, bias) == %Neuron{
             weights: weights,
             bias: bias
             #  sum: 0.0
           }
  end

  test "new_random/1" do
    assert %Neuron{
             bias: bias,
             weights: weights
             #  sum: sum
           } = Neuron.new_random(4)

    # assert is_float(sum)
    # assert sum == 0.0
    assert length(weights) == 4
    assert Enum.all?(weights, &is_float/1)
    assert is_float(bias)
  end

  test "feedforward/2" do
    weights = [1.0, 0.0, -1.1]
    bias = 1.0
    n1 = Neuron.new(weights, bias)
    inputs = [1.0, 0.9, 0.0]
    assert 2.0 = Neuron.feedforward(n1, inputs)
  end

  test "backprop/5" do
    weights = [1.0, 0.0, -1.1]
    bias = 1.0
    n1 = Neuron.new(weights, bias)
    assert n1 == %Neuron{weights: weights, bias: bias}
    input = [1.0, 0.9, 0.0]
    output = Neuron.feedforward(n1, input)

    negative_gradient = 0.5
    neuron_local_error = 0.3
    learn_rate = 0.05
    activation_deriv = &Activation.sigmoid_deriv/1
    sum_deriv = activation_deriv.(output)

    assert {next_error, n2} =
             Neuron.backprop(
               n1,
               input,
               sum_deriv,
               negative_gradient,
               neuron_local_error,
               learn_rate
             )

    assert n2 == %Neuron{
             bias: 0.9992125481094737,
             weights: [0.9992125481094737, -7.087067014736697e-4, -1.1]
           }

    assert next_error == [0.10499358540350662, 0.0, -0.11549294394385728]
  end
end
