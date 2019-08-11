defmodule Annex.Layer.DropoutTest do
  use Annex.LayerCase

  alias Annex.{
    AnnexError,
    Layer,
    Layer.Dropout,
    LayerConfig
  }

  describe "init_layer/1" do
    test "ok for valid config" do
      assert Dropout
             |> LayerConfig.build(frequency: 0.5)
             |> Layer.init_layer() == %Dropout{frequency: 0.5}
    end

    test "raises for invalid config" do
      assert_raise(AnnexError, fn ->
        Dropout
        |> LayerConfig.build(frequency: 1.5)
        |> Layer.init_layer()
      end)
    end

    test "works for frequency above 0.0 and less than or equal to 1.0" do
      assert %Dropout{frequency: 1.0} = build(Dropout, frequency: 1.0)
      assert %Dropout{frequency: 0.444} = build(Dropout, frequency: 0.444)
      assert %Dropout{frequency: 0.0} = build(Dropout, frequency: 0.0)
    end

    test "raises for non-frequency" do
      assert_raise(AnnexError, fn -> build(Dropout, frequency: 1.1) end)
      assert_raise(AnnexError, fn -> build(Dropout, frequency: 1) end)
      assert_raise(AnnexError, fn -> build(Dropout, frequency: :one) end)
      assert_raise(AnnexError, fn -> build(Dropout, frequency: -1.0) end)
    end
  end

  describe "feedforward/2" do
    test "works with a list of floats" do
      layer1 = build(Dropout, frequency: 0.5)
      original = 0.666
      {_layer2, pred} = Layer.feedforward(layer1, [original])
      assert [zeroed_or_original] = pred
      assert zeroed_or_original in [original, 0.0]
    end

    test "dropout does not change on feedforward" do
      layer1 = build(Dropout, frequency: 0.5)
      {layer2, _pred} = Layer.feedforward(layer1, [1.0])
      assert layer1 == layer2
    end
  end
end
