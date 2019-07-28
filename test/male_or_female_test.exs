defmodule Annex.SequenceMOrFTest do
  use ExUnit.Case, async: true
  alias Annex.Layer.Sequence

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

    labels = [
      # Alice
      [1.0],
      # Bob
      [0.0],
      # Charlie
      [0.0],
      # Diana
      [1.0]
    ]

    assert {:ok, %Sequence{} = seq, _loss} =
             [
               Annex.dense(2, 2),
               Annex.activation(:sigmoid),
               Annex.dense(1, 2),
               Annex.activation(:sigmoid)
             ]
             |> Annex.sequence()
             |> Annex.train(data, labels,
               name: "male or female based on normalized weight and height",
               halt_condition: {:epochs, 8_000},
               log_interval: 1_000
             )

    [alice_pred] = Annex.predict(seq, [-2.0, -1.0])
    [bob_pred] = Annex.predict(seq, [25.0, 6.0])

    assert_in_delta(alice_pred, 1.0, 0.1)
    assert_in_delta(bob_pred, 0.0, 0.1)
  end
end
