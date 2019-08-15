defmodule Annex.DefaultsTest do
  use ExUnit.Case

  alias Annex.{
    Cost.MeanSquaredError,
    Defaults
  }

  describe "get_defaults/0" do
    test "returns a list" do
      assert Defaults.get_defaults() == [
               learning_rate: 0.05,
               cost: Annex.Cost.MeanSquaredError
             ]
    end
  end

  describe "get_defaults/1" do
    test "returns the value when key exists" do
      assert Defaults.get_defaults(:cost) == MeanSquaredError
    end

    test "returns nil when the key does not exist" do
      assert Defaults.get_defaults(:blep) == nil
    end
  end

  describe "get_defaults/2" do
    test "returns the value when key exists" do
      assert Defaults.get_defaults(:cost, :other) == MeanSquaredError
    end

    test "returns provided default when the key does not exist" do
      assert Defaults.get_defaults(:blep, :other) == :other
    end
  end

  describe "cost/1" do
    test "returns expected module" do
      assert Defaults.cost() == MeanSquaredError
    end
  end

  describe "learning_rate/0" do
    test "returns expected float" do
      assert Defaults.learning_rate() == 0.05
    end
  end
end
