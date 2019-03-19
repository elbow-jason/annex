defmodule Annex.SequenceXorTest do
  use ExUnit.Case, async: true
  alias Annex
  alias Annex.Sequence

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

    seq1 =
      Annex.sequence(
        [
          Annex.dense(8, input_dims: 2),
          Annex.activation(:tanh),
          Annex.dense(1, input_dims: 8),
          Annex.activation(:sigmoid)
        ],
        learning_rate: 0.05
      )

    %Sequence{} =
      seq2 = Annex.train(seq1, data, labels, name: "xor", epochs: 10_000, print_at_epoch: 10_000)

    [zero_zero] = Annex.predict(seq2, [0.0, 0.0])
    [zero_one] = Annex.predict(seq2, [0.0, 1.0])
    [one_zero] = Annex.predict(seq2, [1.0, 0.0])
    [one_one] = Annex.predict(seq2, [1.0, 1.0])

    assert_in_delta(zero_one, 1.0, 0.1)
    assert_in_delta(zero_zero, 0.0, 0.1)
    assert_in_delta(one_zero, 1.0, 0.1)
    assert_in_delta(one_one, 0.0, 0.1)
  end
end
