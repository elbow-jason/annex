defmodule AnnexTest do
  use ExUnit.Case
  alias Annex.{Network, Layer}

  doctest Annex

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
        Layer.new_random(8, 2, activation: :sigmoid),
        Layer.new_random(1, 8, activation: :sigmoid)
      ]
    }

    %Network{} = net2 = Network.train(net1, data, labels, epochs: 20_000, print_at_epoch: 10_000)

    [zero_zero] = Network.predict(net2, [0.0, 0.0])
    [zero_one] = Network.predict(net2, [0.0, 1.0])
    [one_zero] = Network.predict(net2, [1.0, 0.0])
    [one_one] = Network.predict(net2, [1.0, 1.0])

    assert_in_delta(zero_one, 1.0, 0.1)
    assert_in_delta(zero_zero, 0.0, 0.1)
    assert_in_delta(one_zero, 1.0, 0.1)
    assert_in_delta(one_one, 0.0, 0.1)
  end

  test "males vs females based on weight and height" do
    data = [
      # Alice
      [-2.0, -1.0],
      # Bob
      [25.0, 6.0],
      # Charlie
      [17.0, 4.0],
      # Diana
      [-15.0, -6.0]
    ]

    all_y_trues = [
      # Alice
      [1.0],
      # Bob
      [0.0],
      # Charlie
      [0.0],
      # Diana
      [1.0]
    ]

    net1 = %Network{
      layers: [
        Layer.new_random(2, 2, activation: :sigmoid),
        Layer.new_random(1, 2, activation: :sigmoid)
      ]
    }

    %Network{} =
      net2 = Network.train(net1, data, all_y_trues, epochs: 10000, print_at_epoch: 2500)

    [alice_pred] = Network.predict(net2, [-2.0, -1.0])
    [bob_pred] = Network.predict(net2, [25.0, 6.0])

    assert_in_delta(alice_pred, 1.0, 0.1)
    assert_in_delta(bob_pred, 0.0, 0.1)

    # IO.puts("""
    #   Males are 0.0
    #   Females are 1.0
    #   - - -
    #   Predicted Alice => #{alice_pred}
    #   Predicted Bob   => #{bob_pred}
    # """)
  end
end
