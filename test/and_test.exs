defmodule Annex.AndTest do
  use ExUnit.Case
  alias Annex.Layer.Sequence
  alias Annex.LearnerHelper

  @dataset [
    {[1.0, 1.0, 1.0], [1.0]},
    {[1.0, 0.0, 1.0], [0.0]},
    {[0.0, 0.0, 0.0], [0.0]},
    {[1.0, 0.0, 0.0], [0.0]},
    {[0.0, 0.0, 1.0], [0.0]},
    {[0.0, 1.0, 0.0], [0.0]},
    {[0.0, 1.0, 1.0], [0.0]}
  ]

  test "and works" do
    seq_config =
      Annex.sequence([
        Annex.dense(11, 3),
        Annex.activation(:tanh),
        Annex.dense(1, 11),
        Annex.activation(:tanh)
      ])

    assert {%Sequence{} = seq, _training_output} =
             Annex.train(seq_config, @dataset,
               learning_rate: 0.15,
               name: "AND operation",
               halt_condition: {:epochs, 8000},
               log: &LearnerHelper.test_logger/4,
               log_interval: 1_000
             )

    [should_be_true] = Annex.predict(seq, [1.0, 1.0, 1.0])
    [should_be_false] = Annex.predict(seq, [1.0, 0.0, 1.0])

    assert_in_delta(should_be_true, 1.0, 0.1)
    assert_in_delta(should_be_false, 0.0, 0.1)
  end
end
