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
    dense = %Dense{neurons: [n1, n2]} = fixture()
    inputs = [0.9, 0.01, 0.1]
    {output, new_dense} = Dense.feedforward(dense, inputs)
    assert output == [0.71243, 1.536]
    n1 = %Neuron{n1 | inputs: inputs, sum: 0.71243}
    n2 = %Neuron{n2 | inputs: inputs, sum: 1.536}

    expected_dense = %Dense{dense | neurons: [n1, n2]}

    assert new_dense == expected_dense
  end

  test "dense backprop works" do
    dense = %Dense{neurons: [n1, n2]} = fixture()
    assert n1 == %Neuron{weights: [-0.3333, 0.24, 0.1]}
    assert n2 == %Neuron{weights: [0.7, -0.4, -0.9]}

    inputs = [0.1, 1.0, 0.0]
    labels = [1.0, 0.0]
    {outputs, dense} = Dense.feedforward(dense, inputs)

    assert outputs == [1.20667, 0.6699999999999999]

    total_loss_pd = Sequence.total_loss_pd(outputs, labels)
    assert total_loss_pd == 1.7533399999999997
    ones = Enum.map(labels, fn _ -> 1.0 end)

    backprop =
      Backprop.new()
      |> Backprop.put_net_loss(total_loss_pd)
      |> Backprop.put_loss_pds(ones)
      |> Backprop.put_derivative(&Activation.sigmoid_deriv/1)
      |> Backprop.put_learning_rate(0.05)
      |> Backprop.put_cost_func(&Cost.mse/1)

    assert dense == %Dense{
             cols: 3,
             neurons: [
               %Neuron{
                 bias: 1.0,
                 inputs: [0.1, 1.0, 0.0],
                 output: 0.0,
                 sum: 1.20667,
                 weights: [-0.3333, 0.24, 0.1]
               },
               %Neuron{
                 bias: 1.0,
                 inputs: [0.1, 1.0, 0.0],
                 output: 0.0,
                 sum: 0.6699999999999999,
                 weights: [0.7, -0.4, -0.9]
               }
             ],
             rows: 2
           }

    assert {new_dense, backprop2} = Dense.backprop(dense, backprop)

    assert backprop2.loss_pds == [
             0.014029189664764501,
             0.00491599279921316,
             0.020463274335004233
           ]

    n1 = %Neuron{
      n1
      | bias: 0.9844604158342636,
        weights: [-0.3348539584165736, 0.22446041583426357, 0.1],
        inputs: inputs,
        sum: 1.20667
    }

    n2 = %Neuron{
      n2
      | bias: 0.9803698920690089,
        weights: [0.6980369892069008, -0.41963010793099104, -0.9],
        inputs: inputs,
        sum: 0.6699999999999999
    }

    assert new_dense == %Dense{dense | neurons: [n1, n2]}
  end
end
