defmodule Annex.SequenceXorTest do
  use ExUnit.Case, async: true

  alias Annex.Layer.Sequence

  test "xor test" do
    dataset = [
      {[0.0, 0.0], [0.0]},
      {[0.0, 1.0], [1.0]},
      {[1.0, 0.0], [1.0]},
      {[1.0, 1.0], [0.0]}
    ]

    assert {%Sequence{} = seq, _training_output} =
             [
               Annex.dense(8, 2),
               Annex.activation(:tanh),
               Annex.dense(1, 8),
               Annex.activation(:sigmoid)
             ]
             |> Annex.sequence()
             |> Annex.train(dataset,
               name: "XOR operation",
               learning_rate: 0.05,
               halt_condition: {:epochs, 8_000},
               log_interval: 1_000
             )

    [zero_zero] = Annex.predict(seq, [0.0, 0.0])
    [zero_one] = Annex.predict(seq, [0.0, 1.0])
    [one_zero] = Annex.predict(seq, [1.0, 0.0])
    [one_one] = Annex.predict(seq, [1.0, 1.0])

    assert_in_delta(zero_one, 1.0, 0.1)
    assert_in_delta(zero_zero, 0.0, 0.1)
    assert_in_delta(one_zero, 1.0, 0.1)
    assert_in_delta(one_one, 0.0, 0.1)
  end
end
