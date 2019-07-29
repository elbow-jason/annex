defmodule Annex.Data.DMatrixTest do
  use ExUnit.Case

  alias Annex.Data.DMatrix

  test "DMatrix enumerates floats with Enum.map/2" do
    dmatrix = DMatrix.cast([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], {2, 3})

    Enum.map(dmatrix, fn item ->
      assert is_float(item) == true, """
      When a DMatrix is enumerated the values should be floats.
      got: #{inspect(item)}
      """
    end)
  end

  describe "mutliply/2" do
    test "works for floats" do
      dmatrix = DMatrix.build([[1.0, 2.0, 3.0]])
      assert DMatrix.multiply(dmatrix, 2.0) == DMatrix.build([[2.0, 4.0, 6.0]])
    end
  end

  describe "dot/2" do
    setup do
      d1 = DMatrix.build([[1.0, 1.0, 1.0]])

      d2 =
        DMatrix.build([
          [2.0, 3.0],
          [2.0, 3.0],
          [2.0, 3.0]
        ])

      {:ok, d1: d1, d2: d2}
    end

    test "produces the correct shape", %{d1: d1, d2: d2} do
      d3 = DMatrix.dot(d1, d2)
      assert DMatrix.shape(d1) == {1, 3}
      assert DMatrix.shape(d2) == {3, 2}
      assert DMatrix.shape(d3) == {1, 2}
    end

    test "produces the correct value", %{d1: d1, d2: d2} do
      assert DMatrix.dot(d1, d2) == DMatrix.build([[6.0, 9.0]])
    end
  end
end
