defmodule Annex.SequenceMOrFTest do
  use ExUnit.Case, async: true
  alias Annex.Layer.Sequence

  test "males vs females based on weight and height" do
    dataset = [
      # Alice
      {[-2.0, -1.0], [1.0]},
      # Bob
      {[25.0, 6.0], [0.0]},
      # Charlie
      {[17.0, 4.0], [0.0]},
      # Diana
      {[-15.0, -6.0], [1.0]}
    ]

    assert {%Sequence{} = seq, _training_output} =
             [
               Annex.dense(2, 2),
               Annex.activation(:tanh),
               Annex.dense(1, 2),
               Annex.activation(:sigmoid)
             ]
             |> Annex.sequence()
             |> Annex.train(dataset,
               name: "male or female based on normalized weight and height",
               halt_condition: {:epochs, 2000},
               log_interval: 1000
             )

    [alice_pred] = Annex.predict(seq, [-2.0, -1.0])
    [bob_pred] = Annex.predict(seq, [25.0, 6.0])

    assert_in_delta(alice_pred, 1.0, 0.1)
    assert_in_delta(bob_pred, 0.0, 0.1)
  end
end
