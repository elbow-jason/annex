defmodule Annex.DenseTest do
  use ExUnit.Case, async: true
  alias Annex.{Dense, Neuron, Sequence}

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
    inputs = [0.1, 1.0, 0.0]
    labels = [1.0, 0.0]
    {outputs, dense} = Dense.feedforward(dense, inputs)
    total_loss_pd = Sequence.total_loss_pd(outputs, labels)
    assert total_loss_pd == 5.08
    ones = Enum.map(labels, fn _ -> 1.0 end)
    assert {backprop_data, [], new_dense} = Dense.backprop(dense, total_loss_pd, ones, [])

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
      | bias: 0.9753125449806411,
        weights: [0.9975312544980641, 0.9753125449806411, 1.0],
        inputs: inputs,
        sum: 2.1
    }

    n2 = %Neuron{
      n2
      | bias: 0.9606666450864929,
        weights: [0.3960666645086493, 0.36066664508649293, 0.4],
        inputs: inputs,
        sum: 1.44
    }

    assert new_dense == %Dense{dense | neurons: [n1, n2]}
  end
end
