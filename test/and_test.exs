defmodule Annex.AndTest do
  use ExUnit.Case
  alias Annex.Sequence

  @data [
    [1.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0]
  ]

  @labels [
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0]
  ]

  test "and works" do
    seq1 =
      Annex.sequence([
        Annex.dense(6, input_dims: 3),
        Annex.activation(:tanh),
        Annex.dense(2, input_dims: 6),
        Annex.activation(:sigmoid)
      ])

    %Sequence{} =
      seq2 =
      Annex.train(seq1, @data, @labels,
        learning_rate: 0.05,
        name: "and T or F",
        halt_condition: {:epochs, 100_000}
      )

    [all_true_sig_true, all_true_sig_false] = Annex.predict(seq2, [1.0, 1.0, 1.0])

    assert_in_delta(all_true_sig_false, 0.0, 0.1)
    assert_in_delta(all_true_sig_true, 1.0, 0.1)
  end
end
