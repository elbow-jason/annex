defmodule Annex.Layer.DenseTest do
  use ExUnit.Case, async: true

  alias Annex.{
    Cost,
    Data,
    Data.DMatrix,
    Layer.Activation,
    Layer.Backprop,
    Layer.Dense,
    Layer.Sequence,
    Utils
  }

  def fixture do
    weights = [-0.3333, 0.24, 0.1, 0.7, -0.4, -0.9]
    biases = [1.0, 1.0]
    Dense.build(2, 3, weights, biases)
  end

  test "dense feedforward works" do
    assert %Dense{} = dense = fixture()
    input = DMatrix.build([[0.9], [0.01], [0.1]])
    output = DMatrix.build([[0.71243], [1.536]])
    assert {new_dense, ^output} = Dense.feedforward(dense, input)
    assert new_dense == %Dense{dense | input: input, output: output}
  end

  test "dense backprop works" do
    dense = %Dense{weights: w} = fixture()
    assert Data.shape(w) == {2, 3}

    assert DMatrix.to_list_of_lists(w) == [
             [-0.3333, 0.24, 0.1],
             [0.7, -0.4, -0.9]
           ]

    # assert n1 == %Neuron{weights: }
    # assert n2 == %Neuron{weights: [0.7, -0.4, -0.9]}

    input = DMatrix.build([[0.1], [1.0], [0.0]])
    labels = [1.0, 0.0]
    {dense, output} = Dense.feedforward(dense, input)

    assert output == DMatrix.build([[1.20667], [0.6699999999999999]])
    assert flat_output = DMatrix.to_flat_list(output)
    error = Sequence.error(flat_output, labels)

    error_matrix =
      error
      |> DMatrix.build()
      |> DMatrix.transpose()

    assert error_matrix == DMatrix.build([[0.2066699999999999], [0.6699999999999999]])

    # gradient = MeanSquaredError.derivative(error, input, labels)
    # negative_gradient = gradient * -1.0
    # assert negative_gradient == 1.7533399999999997

    backprop =
      Backprop.new(
        derivative: &Activation.sigmoid_deriv/1,
        # negative_gradient: negative_gradient,
        learning_rate: 0.05,
        cost_func: &Cost.mse/1
      )

    assert dense == %Dense{dense | input: input, output: output}
    assert {new_dense, next_error, backprop2} = Dense.backprop(dense, error_matrix, backprop)

    # assert next_error == [0.09766197256953889, -0.04702502620848973, -0.18379936260301233]
    assert next_error ==
             DMatrix.build([
               [0.400116889],
               [-0.2183992],
               [-0.582333]
             ])

    # updated_weights = [
    #   -0.33362115658595326,
    #   0.23678843414046724,
    #   0.1,
    #   0.6986847827686236,
    #   -0.413152172313764,
    #   -0.9
    # ]
    # updated_biases = [0.9967884341404672, 0.986847827686236]

    new_weights =
      DMatrix.build([
        [-0.33348316845902864, 0.2381683154097136, 0.1],
        [0.6992498789559489, -0.40750121044051013, -0.9]
      ])

    new_biases =
      DMatrix.build([
        [0.9981683154097136],
        [0.9924987895594899]
      ])

    assert new_dense == %Dense{
             dense
             | input: nil,
               output: nil,
               weights: new_weights,
               biases: new_biases
           }
  end

  @dense_2_by_3 Dense.build(2, 3, [1.0, 1.0, 1.0, 0.5, 0.5, 0.5], [1.0, 1.0])

  describe "build/4" do
    test "outputs the correct shape" do
      built = Dense.build(2, 3, [1.0, 1.0, 1.0, 0.5, 0.5, 0.5], [1.0, 1.0])

      assert built == %Dense{
               weights: DMatrix.build([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]),
               biases: DMatrix.build([[1.0], [1.0]]),
               rows: 2,
               columns: 3,
               initialized?: true
             }

      dense = Dense.build(2, 1, [1.0, 1.0], [1.0, 1.0])
      assert dense.weights == DMatrix.build([[1.0], [1.0]])
      assert Data.shape(DMatrix, dense.weights) == {2, 1}
      assert Dense.shape(dense) == {2, 1}
    end
  end

  describe "improves at prediction" do
    test "when presented with the same input" do
      dense = @dense_2_by_3
      inputs = DMatrix.build([[0.0], [1.0], [0.0]])
      labels = DMatrix.build([[2.0], [0.0]])
      backprops = [learning_rate: 0.01, derivative: fn x -> 1.0 end]
      # iter 1
      {dense1, output1} = Dense.feedforward(dense, inputs)
      assert output1 == DMatrix.build([[2.0], [1.5]])
      error1 = DMatrix.subtract(labels, output1)
      {dense1, _error2, _backprops} = Dense.backprop(dense1, error1, backprops)

      # iter 2
      {dense2, output2} = Dense.feedforward(dense1, inputs)
      assert output2 == DMatrix.build([[2.0], [1.5299999999999998]])
      error2 = DMatrix.subtract(labels, output2)
      {dense2, _error2, _backprops} = Dense.backprop(dense2, error2, backprops)

      # iter 3
      {dense3, output3} = Dense.feedforward(dense2, inputs)
      assert output3 == DMatrix.build([[2.0], [1.5606]])
      error3 = DMatrix.subtract(labels, output3)
      {_dense3, _error2, _backprops} = Dense.backprop(dense3, error3, backprops)
    end
  end

  describe "feedforward/2" do
    setup do
      inputs = DMatrix.build([[1.0], [1.0], [1.0]])
      {:ok, inputs: inputs}
    end

    test "outputs the correct shape", %{inputs: inputs} do
      {_dense2, outputs} = Dense.feedforward(@dense_2_by_3, inputs)
      assert Data.shape(DMatrix, outputs) == {2, 1}
    end

    test "outputs the correct values", %{inputs: inputs} do
      {_dense2, outputs} = Dense.feedforward(@dense_2_by_3, inputs)
      expected_dot1 = 1.0 * 1.0 + 1.0 * 1.0 + 1.0 * 1.0 + 1.0
      expected_dot2 = 0.5 * 1.0 + 0.5 * 1.0 + 0.5 * 1.0 + 1.0
      assert [expected_dot1, expected_dot2] == [4.0, 2.5]
      assert Data.to_flat_list(outputs) == [4.0, 2.5]
    end

    test "updates Dense struct's `:input` field correctly", %{inputs: inputs} do
      {dense2, _outputs} = Dense.feedforward(@dense_2_by_3, inputs)
      assert %Dense{input: ^inputs} = dense2
    end

    test "updates Dense struct's `:output` field correctly", %{inputs: inputs} do
      {dense2, outputs} = Dense.feedforward(@dense_2_by_3, inputs)
      assert %Dense{output: ^outputs} = dense2
    end
  end

  describe "backprop/3" do
    setup do
      inputs = DMatrix.build([[1.0], [1.0], [1.0]])
      {:ok, inputs: inputs, dense: @dense_2_by_3}
    end

    test "nudges weights in the positive direction when the error is positive", %{
      inputs: inputs,
      dense: dense1
    } do
      # setup
      %Dense{weights: weights1} = dense1
      {dense2, outputs} = Dense.feedforward(dense1, inputs)
      label = DMatrix.build([[4.0], [3.0]])

      errors1 = DMatrix.subtract(label, outputs)

      learning_rate = 0.05
      identity_derivative = fn _ -> 1.0 end

      backprops1 =
        Backprop.new(
          derivative: identity_derivative,
          learning_rate: learning_rate
          # cost_func: &Cost.mse/1
        )

      # preconditions
      ## outputs are correct
      assert outputs == DMatrix.build([[4.0], [2.5]])

      ## errors are correct
      assert errors1 == DMatrix.build([[0.0], [0.5]])

      # execute the code
      {dense3, errors2, _backprops2} = Dense.backprop(dense2, errors1, backprops1)

      # post conditions
      # all the columns were the same so the each is equally errorful.
      assert errors2 == DMatrix.build([[0.25], [0.25], [0.25]])
      assert %Dense{weights: weights3} = dense3

      # with the weights adjusted
      # the shape the weights1 should be the same as the shape for weight3
      assert Data.shape(DMatrix, weights1) == Data.shape(DMatrix, weights3)

      # we dont care about vectors. We want lists for easy visual parsing,
      # code comprehension, and simplicity
      # weights3 should bave 2 rows
      assert [row1, row2] = DMatrix.to_list_of_lists(weights3)

      # the output should have a 2 outputs
      assert [4.0, 2.5 = row2_output] = DMatrix.to_flat_list(outputs)
      assert [_row1, weights1_row2] = DMatrix.to_list_of_lists(weights1)
      assert [_row1, weights3_row2] = DMatrix.to_list_of_lists(weights3)

      # the error1 should have 2 outputs
      # Since the errors are [0.0, 0.5]...
      assert [0.0 = _row1_error, 0.5 = row2_error] = DMatrix.to_flat_list(errors1)

      # the first row had 0.0 error and the weights should have remained unchanged
      assert row1 == [1.0, 1.0, 1.0]

      # the second row had error of 0.5 so the gradient to the weights should be
      # row2_output * learning_rate * row2_error
      row2_gradient = -0.025
      expected_weights3_row2 = Enum.map(weights1_row2, fn el -> el + row2_gradient end)
      assert weights3_row2 == expected_weights3_row2
    end

    test "nudges weights closer to zero when the error is negative", %{
      inputs: inputs,
      dense: dense1
    } do
      # setup
      %Dense{weights: weights1} = dense1
      {dense2, outputs} = Dense.feedforward(dense1, inputs)
      label = DMatrix.build([[4.0], [2.0]])

      errors1 = DMatrix.subtract(label, outputs)

      learning_rate = 0.05
      identity_derivative = fn _ -> 1.0 end

      backprops1 =
        Backprop.new(
          derivative: identity_derivative,
          learning_rate: learning_rate
          # cost_func: &Cost.mse/1
        )

      # preconditions
      ## outputs are correct
      assert outputs == DMatrix.build([[4.0], [2.5]])

      ## errors are correct
      assert errors1 == DMatrix.build([[0.0], [-0.5]])

      # execute the code
      {dense3, errors2, _backprops2} = Dense.backprop(dense2, errors1, backprops1)

      # post conditions
      # all the columns were the same so the each is equally errorful.
      assert errors2 == DMatrix.build([[-0.25], [-0.25], [-0.25]])
      assert %Dense{weights: weights3} = dense3

      # with the weights adjusted
      # the shape the weights1 should be the same as the shape for weight3
      assert Data.shape(DMatrix, weights1) == Data.shape(DMatrix, weights3)

      # we dont care about vectors. We want lists for easy visual parsing,
      # code comprehension, and simplicity
      # in weights3 should bave 2 rows

      assert [row1, row2] = DMatrix.to_list_of_lists(weights3)

      # the output should have a 2 outputs
      assert [4.0, 2.5 = row2_output] = DMatrix.to_flat_list(outputs)
      assert [_row1, weights1_row2] = DMatrix.to_list_of_lists(weights1)
      assert [_row1, weights3_row2] = DMatrix.to_list_of_lists(weights3)

      # the error1 should have 2 outputs
      # Since the errors are [0.0, 0.5]...
      assert [0.0 = _row1_error, -0.5 = row2_error] = DMatrix.to_flat_list(errors1)

      # the first row had 0.0 error and the weights should have remained unchanged
      assert row1 == [1.0, 1.0, 1.0]

      # the second row had error of 0.5 so the gradient to the weights should be
      row2_gradient = 0.025

      expected_weights3_row2 = Enum.map(weights1_row2, fn el -> el + row2_gradient end)
      assert weights3_row2 == expected_weights3_row2

      Utils.zipmap(weights1_row2, weights3_row2, fn a, b ->
        diff = a - b

        cond do
          a > 0.0 ->
            assert diff < 0.0

          a < 0.0 ->
            assert diff > 0.0

          a === 0.0 ->
            assert diff == 0.0
        end
      end)
    end
  end
end
