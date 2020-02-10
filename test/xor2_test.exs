defmodule Annex.SequenceXor2Test do
  use ExUnit.Case, async: true
  alias Annex.Layer.Sequence

  test "xor2 test" do
    dataset = [
      {[0.0, 0.0], [0.0, 1.0]},
      {[0.0, 1.0], [1.0, 0.0]},
      {[1.0, 0.0], [1.0, 0.0]},
      {[1.0, 1.0], [0.0, 1.0]}
    ]

    assert {%Sequence{} = seq, _training_output} =
             [
               Annex.dense(11, 2),
               Annex.activation(:relu),
               Annex.dense(2, 11),
               Annex.activation(:sigmoid)
             ]
             |> Annex.sequence()
             |> Annex.train(dataset,
               name: "xor2",
               learning_rate: 0.1,
               halt_condition: {:epochs, 1200},
               log_interval: 600
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
