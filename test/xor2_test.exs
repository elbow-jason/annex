defmodule Annex.SequenceXor2Test do
  use ExUnit.Case, async: true

  test "xor2 test" do
    data = [
      [0.0, 0.0],
      [0.0, 1.0],
      [1.0, 0.0],
      [1.0, 1.0]
    ]

    labels = [
      [0.0, 1.0],
      [1.0, 0.0],
      [1.0, 0.0],
      [0.0, 1.0]
    ]

    {_output, seq} =
      Annex.sequence([
        Annex.dense(16, input_dims: 2),
        Annex.activation(:tanh),
        Annex.dense(2, input_dims: 16),
        Annex.activation(:sigmoid)
      ])
      |> Annex.train(data, labels,
        name: "xor2",
        learning_rate: 0.05,
        halt_condition: {:epochs, 120_000}
      )

    [pred_yes, pred_no] = Annex.predict(seq, [0.0, 0.0])
    assert_in_delta(pred_yes, 0.0, 0.1)
    assert_in_delta(pred_no, 1.0, 0.1)
    [pred_yes, pred_no] = Annex.predict(seq, [0.0, 1.0])
    assert_in_delta(pred_yes, 1.0, 0.1)
    assert_in_delta(pred_no, 0.0, 0.1)
    [pred_yes, pred_no] = Annex.predict(seq, [1.0, 0.0])
    assert_in_delta(pred_yes, 1.0, 0.1)
    assert_in_delta(pred_no, 0.0, 0.1)
    [pred_yes, pred_no] = Annex.predict(seq, [1.0, 1.0])
    assert_in_delta(pred_yes, 0.0, 0.1)
    assert_in_delta(pred_no, 1.0, 0.1)
  end
end
