defmodule Annex.Layer.DropoutTest do
  use Annex.LayerCase

  alias Annex.{
    AnnexError,
    Layer,
    Layer.Dropout,
    LayerConfig
  }

  require Dropout

  describe "is_frequency/1 guard" do
    test "true for float between 0.0 and 1.0" do
      assert Dropout.is_frequency(0.0) == true
      assert Dropout.is_frequency(0.5) == true
      assert Dropout.is_frequency(1.0) == true
    end

    test "false for float below 0.0" do
      assert Dropout.is_frequency(-0.1) == false
    end

    test "false for float above 1.0" do
      assert Dropout.is_frequency(1.1) == false
    end

    test "false for everything else" do
      assert Dropout.is_frequency(:belp) == false
    end

    test "useable as a guard" do
      case Enum.random([0.0, 0.5, 1.0]) do
        x when Dropout.is_frequency(x) -> assert true
        _ -> assert false, "is_frequency failed"
      end
    end
  end

  describe "init_layer/1" do
    test "ok for valid config" do
      assert Dropout
             |> LayerConfig.build(frequency: 0.5)
             |> Layer.init_layer() == %Dropout{frequency: 0.5}
    end

    test "raises for invalid :frequency in config" do
      assert_raise(AnnexError, fn ->
        Dropout
        |> LayerConfig.build(frequency: 1.5)
        |> Layer.init_layer()
      end)
    end

    test "raises for missing :frequency in config" do
      assert_raise(AnnexError, fn ->
        Dropout
        |> LayerConfig.build([])
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
