defmodule Annex.Layer.DenseTest do
  use ExUnit.Case, async: true

  alias Annex.{
    Cost,
    Cost.MeanSquaredError,
    Data.DMatrix,
    Layer.Activation,
    Layer.Backprop,
    Layer.Dense,
    # Layer.Neuron,
    Layer.Sequence
  }

  def fixture do
    weights = [-0.3333, 0.24, 0.1, 0.7, -0.4, -0.9]
    biases = [1.0, 1.0]
    Dense.build(2, 3, weights, biases)
  end

  test "dense feedforward works" do
    assert %Dense{} = dense = fixture()
    input = [0.9, 0.01, 0.1]
    output = [0.71243, 1.536]
    assert {new_dense, ^output} = Dense.feedforward(dense, input)
    assert new_dense == %Dense{dense | input: input, output: output}
  end

  test "dense backprop works" do
    dense = %Dense{weights: w} = fixture()

    assert DMatrix.to_list_of_lists(w) == [
             [-0.3333, 0.24, 0.1],
             [0.7, -0.4, -0.9]
           ]

    # assert n1 == %Neuron{weights: }
    # assert n2 == %Neuron{weights: [0.7, -0.4, -0.9]}

    input = [0.1, 1.0, 0.0]
    labels = [1.0, 0.0]
    {dense, output} = Dense.feedforward(dense, input)

    assert output == DMatrix.build([[1.20667], [0.6699999999999999]])
    assert flat_output = DMatrix.to_flat_list(output)
    error = Sequence.error(flat_output, labels)
    error_matrix = DMatrix.build(error)

    gradient = MeanSquaredError.derivative(error, input, labels)
    negative_gradient = gradient * -1.0
    assert negative_gradient == 1.7533399999999997

    backprop =
      Backprop.new(
        derivative: &Activation.sigmoid_deriv/1,
        negative_gradient: negative_gradient,
        learning_rate: 0.05,
        cost_func: &Cost.mse/1
      )

    assert dense == %Dense{dense | input: input, output: output}

    assert {new_dense, next_error, backprop2} = Dense.backprop(dense, error_matrix, backprop)

    assert next_error == [0.09766197256953889, -0.04702502620848973, -0.18379936260301233]

    updated_weights = [
      -0.33362115658595326,
      0.23678843414046724,
      0.1,
      0.6986847827686236,
      -0.413152172313764,
      -0.9
    ]

    updated_biases = [0.9967884341404672, 0.986847827686236]

    assert new_dense == %Dense{
             dense
             | weights: DMatrix.build(updated_weights, 3, 2),
               biases: DMatrix.build(updated_biases, 1, 2)
           }
  end
end
