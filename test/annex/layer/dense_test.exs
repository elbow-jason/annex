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
    input = [0.9, 0.01, 0.1]
    {new_dense, output} = Dense.feedforward(dense, input)
    assert output == [0.71243, 1.536]
    n1 = %Neuron{n1 | inputs: input, sum: 0.71243}
    n2 = %Neuron{n2 | inputs: input, sum: 1.536}

    expected_dense = %Dense{dense | neurons: [n1, n2], input: input, output: output}

    assert new_dense == expected_dense
  end

  test "dense backprop works" do
    dense = %Dense{neurons: [n1, n2]} = fixture()
    assert n1 == %Neuron{weights: [-0.3333, 0.24, 0.1]}
    assert n2 == %Neuron{weights: [0.7, -0.4, -0.9]}

    input = [0.1, 1.0, 0.0]
    labels = [1.0, 0.0]
    {dense, output} = Dense.feedforward(dense, input)

    assert output == [1.20667, 0.6699999999999999]

    total_loss_pd = Sequence.total_loss_pd(output, labels)
    assert total_loss_pd == 1.7533399999999997
    loss_pds = Enum.map(labels, fn _ -> 1.0 end)

    backprop = Backprop.new(net_loss: total_loss_pd, learning_rate: 0.05, cost_func: &Cost.mse/1)

    assert dense == %Dense{
             input: input,
             output: output,
             cols: 3,
             neurons: [
               %Neuron{
                 bias: 1.0,
                 inputs: input,
                 #  output: 0.0,
                 sum: 1.20667,
                 weights: [-0.3333, 0.24, 0.1]
               },
               %Neuron{
                 bias: 1.0,
                 inputs: input,
                 #  output: 0.0,
                 sum: 0.6699999999999999,
                 weights: [0.7, -0.4, -0.9]
               }
             ],
             rows: 2
           }

    assert {new_dense, next_loss_pds, backprop2} = Dense.backprop(dense, loss_pds, backprop)

    assert next_loss_pds == [
             0.014029189664764501,
             0.00491599279921316,
             0.020463274335004233
           ]

    n1 = %Neuron{
      n1
      | bias: 0.9844604158342636,
        weights: [-0.3348539584165736, 0.22446041583426357, 0.1],
        inputs: input,
        sum: 1.20667
    }

    n2 = %Neuron{
      n2
      | bias: 0.9803698920690089,
        weights: [0.6980369892069008, -0.41963010793099104, -0.9],
        inputs: input,
        sum: 0.6699999999999999
    }

    assert new_dense == %Dense{dense | neurons: [n1, n2]}
  end
end
