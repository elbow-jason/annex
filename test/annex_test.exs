defmodule AnnexTest do
  use ExUnit.Case
  doctest Annex

  alias Annex.{
    Data.DMatrix,
    Layer.Activation,
    Layer.Dense,
    Layer.Dropout,
    Layer.Sequence,
    LayerConfig
  }

  describe "sequence/1" do
    test "works" do
      assert Annex.sequence([
               Annex.dense(1, 1)
             ]) == %LayerConfig{
               module: Sequence,
               details: %{
                 layers: [
                   %LayerConfig{
                     module: Dense,
                     details: %{
                       rows: 1,
                       columns: 1
                     }
                   }
                 ]
               }
             }
    end
  end

  describe "dropout/1" do
    test "works" do
      assert Annex.dropout(0.4) == %LayerConfig{
               module: Dropout,
               details: %{
                 frequency: 0.4
               }
             }
    end
  end

  describe "dense/4" do
    test "works" do
      weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
      biases = [0.1, 0.1, 0.1]

      assert Annex.dense(2, 3, weights, biases) == %LayerConfig{
               module: Dense,
               details: %{
                 rows: 2,
                 columns: 3,
                 weights: weights,
                 biases: biases
               }
             }
    end
  end

  describe "dense/2" do
    test "works" do
      assert Annex.dense(2, 3) == %LayerConfig{
               module: Dense,
               details: %{
                 rows: 2,
                 columns: 3
               }
             }
    end
  end

  describe "activation/1" do
    test "works" do
      assert Annex.activation(:relu) == %LayerConfig{
               module: Activation,
               details: %{
                 name: :relu
               }
             }
    end
  end

  describe "train/2" do
    test "works" do
      weights = [0.1, 0.15, 0.17, 0.2, 0.24, 0.28]
      biases = [0.1, 0.1]

      dataset = [
        {[1.0, 0.4, 1.4], [1.0, 0.0]}
      ]

      assert {seq, training_output} =
               [
                 Annex.dense(2, 3, weights, biases),
                 Annex.activation(:relu)
               ]
               |> Annex.sequence()
               |> Annex.train(dataset, halt_condition: {:epochs, 1})

      assert %Sequence{layers: %{0 => dense, 1 => activation}} = seq

      assert dense == %Dense{
               rows: 2,
               columns: 3,
               data_type: DMatrix,
               biases: DMatrix.build([[0.12510000000000002], [0.0606]]),
               weights:
                 DMatrix.build([
                   [0.12510000000000002, 0.16004, 0.20514000000000002],
                   [0.16060000000000002, 0.22424, 0.22484000000000004]
                 ])
             }

      assert %Activation{name: :relu} = activation
    end
  end

  describe "predict/2" do
    test "works" do
      weights = [0.1, 0.15, 0.17, 0.2, 0.24, 0.28]
      biases = [0.1, 0.1]

      dataset = [
        {[1.0, 0.4, 1.4], [1.0, 0.0]}
      ]

      assert {seq, training_output} =
               [
                 Annex.dense(2, 3, weights, biases),
                 Annex.activation(:relu)
               ]
               |> Annex.sequence()
               |> Annex.train(dataset, halt_condition: {:epochs, 1})

      assert Annex.predict(seq, [1.0, 0.4, 1.4]) == [0.6014120000000001, 0.625672]
    end
  end
end
