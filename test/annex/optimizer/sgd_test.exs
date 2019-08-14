defmodule Annex.Optimizer.SGDTest do
  use ExUnit.Case

  alias Annex.{
    AnnexError,
    Layer.Sequence,
    Optimizer.SGD
  }

  describe "train/3" do
    test "trains with SGD for Layer learner" do
      seq =
        Annex.sequence([
          Annex.dense(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 1.0, 1.0]),
          Annex.activation(:relu)
        ])
        |> Sequence.init_layer()

      dataset = [
        {[2.0, 0.0], [1.0, 1.0, 1.0]},
        {[10.0, 100.0], [0.0, 1.0, 0.0]},
        {[100.0, 0100.0], [1.0, 0.0, 1.0]}
      ]

      assert {%Sequence{} = _seq2, %{} = _training_outputs} = SGD.train(seq, dataset, [])
    end

    test "raises for non-Layer" do
      dataset = [
        {[2.0, 0.0], [1.0, 1.0, 1.0]},
        {[10.0, 100.0], [0.0, 1.0, 0.0]},
        {[100.0, 0100.0], [1.0, 0.0, 1.0]}
      ]

      assert_raise(AnnexError, fn -> SGD.train(%URI{}, dataset, []) end)
    end
  end

  describe "batch_dataset/1" do
    test "returns a shuffled dataset of the same size for nil" do
      dataset = [
        {[2.0, 0.0], [1.0, 1.0, 1.0]},
        {[10.0, 100.0], [0.0, 1.0, 0.0]},
        {[100.0, 0100.0], [1.0, 0.0, 1.0]}
      ]

      batch = SGD.batch_dataset(dataset, nil)
      assert length(batch) == length(dataset)
    end

    test "returns a shuffled dataset of n size for datasets larger than n" do
      dataset = [
        {[2.0, 0.0], [1.0, 1.0, 1.0]},
        {[10.0, 100.0], [0.0, 1.0, 0.0]},
        {[100.0, 0100.0], [1.0, 0.0, 1.0]}
      ]

      batch = SGD.batch_dataset(dataset, 2)
      assert length(batch) == 2
    end

    test "returns a shuffled dataset of n size for datasets smaller than n" do
      dataset = [
        {[2.0, 0.0], [1.0, 1.0, 1.0]},
        {[10.0, 100.0], [0.0, 1.0, 0.0]},
        {[100.0, 0100.0], [1.0, 0.0, 1.0]}
      ]

      batch = SGD.batch_dataset(dataset, 10)
      assert length(batch) == 10
    end
  end
end
