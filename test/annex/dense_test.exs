defmodule Annex.DenseTest do
  use ExUnit.Case, async: true
  alias Annex.{Dense, Neuron}

  def fixture() do
    %Dense{
      neurons: [
        Neuron.new([1.0, 1.0, 1.0], 1.0),
        Neuron.new([0.4, 0.4, 0.4], 1.0)
      ],
      rows: 2,
      cols: 3,
      activation_derivative: &Annex.Activation.sigmoid_deriv/1,
      learning_rate: 0.05
    }
  end

  test "dense feedforward works" do
    dense = %Dense{neurons: [n1, n2]} = fixture()
    inputs = [0.3, 0.3, 0.3]
    {output, new_dense} = Dense.feedforward(dense, inputs)
    assert output == [1.9, 1.3599999999999999]
    n1 = %Neuron{n1 | inputs: inputs, sum: 1.9}
    n2 = %Neuron{n2 | inputs: inputs, sum: 1.3599999999999999}

    expected_dense = %Dense{dense | neurons: [n1, n2]}

    assert new_dense == expected_dense
  end

  test "dense backprop works" do
    dense = %Dense{neurons: [n1, n2]} = fixture()
    inputs = [0.3, 0.3, 0.3]
    n1 = %Neuron{n1 | inputs: inputs, sum: 1.9}
    n2 = %Neuron{n2 | inputs: inputs, sum: 1.3599999999999999}
    dense = %Dense{dense | neurons: [n1, n2]}

    assert {backprop_data, [], new_dense} = Dense.backprop(dense, 0.25, [0.2, 0.8], [])

    assert backprop_data = [
             0.11318025926193101,
             0.11318025926193101,
             0.11318025926193101,
             0.06501048048279782,
             0.06501048048279782,
             0.06501048048279782
           ]

    n1 = %Neuron{
      n1
      | bias: 0.9997170493518451,
        weights: [0.9999151148055535, 0.9999151148055535, 0.9999151148055535]
    }

    n2 = %Neuron{
      n2
      | bias: 0.9983747379879301,
        weights: [0.39951242139637905, 0.39951242139637905, 0.39951242139637905]
    }

    assert new_dense == %Dense{dense | neurons: [n1, n2]}
  end
end
