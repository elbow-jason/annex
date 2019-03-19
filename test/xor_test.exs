defmodule Annex.XorTest do
  use ExUnit.Case, async: true
  alias Annex.{Network, Layer}

  test "xor test" do
    data = [
      [0.0, 0.0],
      [0.0, 1.0],
      [1.0, 0.0],
      [1.0, 1.0]
    ]

    labels = [
      [0.0],
      [1.0],
      [1.0],
      [0.0]
    ]

    net1 = %Network{
      layers: [
        Layer.new_random(8, 2, :sigmoid),
        Layer.new_random(1, 8, :sigmoid)
      ]
    }

    %Network{} =
      net2 =
      Network.train(net1, data, labels, name: "xor", epochs: 50_000, print_at_epoch: 50_000)

    [zero_zero] = Network.predict(net2, [0.0, 0.0])
    [zero_one] = Network.predict(net2, [0.0, 1.0])
    [one_zero] = Network.predict(net2, [1.0, 0.0])
    [one_one] = Network.predict(net2, [1.0, 1.0])

    assert_in_delta(zero_one, 1.0, 0.1)
    assert_in_delta(zero_zero, 0.0, 0.1)
    assert_in_delta(one_zero, 1.0, 0.1)
    assert_in_delta(one_one, 0.0, 0.1)
  end
end
