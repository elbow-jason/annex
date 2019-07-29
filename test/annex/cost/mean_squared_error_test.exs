defmodule Annex.Cost.MeanSquaredErrorTest do
  use ExUnit.Case
  alias Annex.Cost.MeanSquaredError

  describe "calculate/1" do
    test "returns 1 for 1s" do
      assert MeanSquaredError.calculate([1.0, 1.0, 1.0]) == 1.0
    end

    test "returns 0 for 0s" do
      assert MeanSquaredError.calculate([0.0, 0.0, 0.0]) == 0.0
    end

    test "correctly calculates MSE 1" do
      assert MeanSquaredError.calculate([3.0]) == 9.0
    end

    test "correctly calculates MSE 2" do
      assert MeanSquaredError.calculate([3.0, 5.0]) == 34.0 / 2.0
    end
  end
end
