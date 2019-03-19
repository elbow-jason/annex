defmodule Annex.SequenceMOrFTest do
  use ExUnit.Case, async: true
  alias Annex.{Sequence}

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

    assert seq =
             Annex.sequence([
               Annex.dense(2, input_dims: 2),
               Annex.activation(:sigmoid),
               Annex.dense(1, input_dims: 2),
               Annex.activation(:sigmoid)
             ])
             |> Annex.train(data, all_y_trues,
               name: "male_or_female",
               epochs: 40000,
               print_at_epoch: 10000
             )

    [alice_pred] = Annex.predict(seq, [-2.0, -1.0])
    [bob_pred] = Annex.predict(seq, [25.0, 6.0])

    assert_in_delta(alice_pred, 1.0, 0.03)
    assert_in_delta(bob_pred, 0.0, 0.03)
  end
end
