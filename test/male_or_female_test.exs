defmodule Annex.MaleOrFemaleText do
  use ExUnit.Case, async: true
  alias Annex.{Network, Layer}

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
        Layer.new_random(2, 2, :sigmoid),
        Layer.new_random(1, 2, :sigmoid)
      ]
    }

    %Network{} =
      net2 =
      Network.train(net1, data, all_y_trues,
        name: "male_or_female",
        epochs: 10000,
        print_at_epoch: 10000
      )

    [alice_pred] = Network.predict(net2, [-2.0, -1.0])
    [bob_pred] = Network.predict(net2, [25.0, 6.0])

    assert_in_delta(alice_pred, 1.0, 0.1)
    assert_in_delta(bob_pred, 0.0, 0.1)
  end
end
