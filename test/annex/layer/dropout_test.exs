defmodule Annex.Layer.DropoutTest do
  use ExUnit.Case
  alias Annex.Layer.Dropout

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
  end
end
