defmodule Annex.Layer.DenseTest do
  use ExUnit.Case, async: true

  alias Annex.{
    Cost,
    Layer.Activation,
    Layer.Backprop,
    Layer.Dense,
    Layer.Neuron,
    Layer.Sequence
  }

  def fixture() do
    %Dense{
      rows: 2,
      cols: 3,
      neurons: [
        Neuron.new([-0.3333, 0.24, 0.1], 1.0),
        Neuron.new([0.7, -0.4, -0.9], 1.0)
      ]
    }
  end

  test "dense feedforward works" do
    assert %Dense{} = dense = fixture()
    input = [0.9, 0.01, 0.1]
    output = [0.71243, 1.536]
    assert {new_dense, ^output} = Dense.feedforward(dense, input)
    assert new_dense == %Dense{dense | input: input, output: output}
  end

  test "dense backprop works" do
    dense = %Dense{neurons: [n1, n2]} = fixture()
    assert n1 == %Neuron{weights: [-0.3333, 0.24, 0.1]}
    assert n2 == %Neuron{weights: [0.7, -0.4, -0.9]}

    input = [0.1, 1.0, 0.0]
    labels = [1.0, 0.0]
    {dense, output} = Dense.feedforward(dense, input)

    assert output == [1.20667, 0.6699999999999999]
    error = Sequence.error(output, labels)
    gradient = Annex.Cost.MeanSquaredError.derivative(error, input, labels)
    negative_gradient = gradient * -1.0
    assert negative_gradient == 1.7533399999999997

    backprop =
      Backprop.new(
        derivative: &Activation.sigmoid_deriv/1,
        negative_gradient: negative_gradient,
        learning_rate: 0.05,
        cost_func: &Cost.mse/1
      )

    assert dense == %Dense{
             input: input,
             output: output,
             cols: 3,
             neurons: [
               %Neuron{
                 bias: 1.0,
                 weights: [-0.3333, 0.24, 0.1]
               },
               %Neuron{
                 bias: 1.0,
                 weights: [0.7, -0.4, -0.9]
               }
             ],
             rows: 2
           }

    assert {new_dense, next_error, backprop2} = Dense.backprop(dense, error, backprop)

    assert next_error == [0.09766197256953889, -0.04702502620848973, -0.18379936260301233]

    n1 = %Neuron{
      n1
      | bias: 0.9967884341404672,
        weights: [-0.33362115658595326, 0.23678843414046724, 0.1]
    }

    n2 = %Neuron{
      n2
      | bias: 0.986847827686236,
        weights: [0.6986847827686236, -0.413152172313764, -0.9]
    }

    assert new_dense == %Dense{dense | neurons: [n1, n2]}
  end
end
