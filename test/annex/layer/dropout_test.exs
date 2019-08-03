defmodule Annex.Layer.DropoutTest do
  use Annex.LayerCase

  alias Annex.{
    AnnexError,
    Layer,
    LayerConfig,
    Layer.Dropout
  }

  describe "init_layer/1" do
    test "ok for valid config" do
      cfg = LayerConfig.build(Dropout, frequency: 0.5)
      {:ok, layer} = Layer.init_layer(cfg)
      assert %Dropout{frequency: 0.5} == layer
    end

    test "error for invalid config" do
      cfg = LayerConfig.build(Dropout, frequency: 1.5)
      {:error, error} = Layer.init_layer(cfg)
      message = "Dropout.build/1 requires a :frequency that is a float between 0.0 and 1.0"

      assert error == %AnnexError{
               message: message,
               details: [
                 invalid_frequency: 1.5,
                 reason: :invalid_frequency_value
               ]
             }
    end

    test "works for frequency above 0.0 and less than or equal to 1.0" do
      assert {:ok, _} = build(Dropout, frequency: 1.0)
      assert {:ok, _} = build(Dropout, frequency: 0.444)
      assert {:ok, _} = build(Dropout, frequency: 0.0)
    end

    test "errors for non-frequency" do
      assert {:error, %AnnexError{}} = build(Dropout, frequency: 1.1)
      assert {:error, %AnnexError{}} = build(Dropout, frequency: 1)
      assert {:error, %AnnexError{}} = build(Dropout, frequency: :one)
      assert {:error, %AnnexError{}} = build(Dropout, frequency: -1.0)
    end
  end

  describe "feedforward/2" do
    test "works with a list of floats" do
      layer1 = build!(Dropout, frequency: 0.5)
      original = 0.666
      {_layer2, pred} = Layer.feedforward(layer1, [original])
      assert [zeroed_or_original] = pred
      assert zeroed_or_original in [original, 0.0]
    end

    test "dropout does not change on feedforward" do
      layer1 = build!(Dropout, frequency: 0.5)
      {layer2, _pred} = Layer.feedforward(layer1, [1.0])
      assert layer1 == layer2
    end
  end
end
