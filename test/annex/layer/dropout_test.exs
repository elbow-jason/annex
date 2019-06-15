defmodule Annex.Layer.DropoutTest do
  use ExUnit.Case

  alias Annex.{
    Layer.Dropout,
    Layer
  }

  describe "build/1" do
    test "works for frequency above 0.0 and less than or equal to 1.0" do
      assert %Dropout{frequency: 1.0} = Dropout.build(1.0)
      assert %Dropout{frequency: 0.444} = Dropout.build(0.444)
      assert %Dropout{frequency: 0.0} = Dropout.build(0.0)
    end

    test "raises for non-frequency" do
      assert_raise(FunctionClauseError, fn -> Dropout.build(1.1) end)
      assert_raise(FunctionClauseError, fn -> Dropout.build(1) end)
      assert_raise(FunctionClauseError, fn -> Dropout.build(:one) end)
      assert_raise(FunctionClauseError, fn -> Dropout.build(-1.0) end)
    end

    test "dropout can handle a list of floats" do
      layer1 = Dropout.build(0.5)
      original = 0.666
      {_layer2, pred} = Layer.feedforward(layer1, [original])
      assert [zeroed_or_original] = pred
      assert zeroed_or_original in [original, 0.0]
    end

    test "dropout does not change on feedforward" do
      layer1 = Dropout.build(0.5)
      {layer2, _pred} = Layer.feedforward(layer1, [1.0])
      assert layer1 == layer2
    end

    test "dropout does not change on init_layer" do
      layer1 = Dropout.build(0.5)
      {:ok, layer2} = Layer.init_layer(layer1)
      assert layer1 == layer2
    end
  end
end
