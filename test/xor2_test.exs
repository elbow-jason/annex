defmodule Annex.SequenceXor2Test do
  use ExUnit.Case, async: true
  alias Annex.Layer.Sequence

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

    {:ok, _output, %Sequence{} = seq} =
      Annex.sequence([
        Annex.dense(11, input_dims: 2),
        Annex.activation(:relu),
        Annex.dense(2, input_dims: 11),
        Annex.activation(:sigmoid)
      ])
      |> Annex.train(data, labels,
        name: "xor2",
        learning_rate: 0.02,
        halt_condition: {:epochs, 200_000}
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
