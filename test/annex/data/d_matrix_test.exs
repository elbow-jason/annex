defmodule Annex.Data.DMatrixTest do
  use ExUnit.Case

  alias Annex.{
    AnnexError,
    Data.DMatrix
  }

  @dmatrix_2_by_3 DMatrix.build([
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]
                  ])

  test "DMatrix enumerates floats with Enum.map/2" do
    Enum.map(@dmatrix_2_by_3, fn item ->
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

  describe "cast/2" do
    test "works for flat data given 1 dimensional shape" do
      flat_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      assert DMatrix.cast(flat_data, [6]) == DMatrix.build([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    end

    test "works for flat data given 2 dimensional shape" do
      flat_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

      assert DMatrix.cast(flat_data, [3, 2]) ==
               DMatrix.build([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

      assert DMatrix.cast(flat_data, [6, 1]) ==
               DMatrix.build([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])

      assert DMatrix.cast(flat_data, [1, 6]) ==
               DMatrix.build([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    end

    test "raises for flat data given invalid shape" do
      assert_raise(AnnexError, fn -> DMatrix.cast(@dmatrix_2_by_3, [7, 1]) end)
    end

    test "works for DMatrix given 2 dimensional shape" do
      assert DMatrix.cast(@dmatrix_2_by_3, [3, 2]) ==
               DMatrix.build([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

      assert DMatrix.cast(@dmatrix_2_by_3, [6, 1]) ==
               DMatrix.build([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])

      assert DMatrix.cast(@dmatrix_2_by_3, [1, 6]) ==
               DMatrix.build([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    end

    test "works for DMatrix given 1 dimensional shape" do
      built = DMatrix.build([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
      assert DMatrix.cast(@dmatrix_2_by_3, [6]) == built
      assert DMatrix.shape(built) == [1, 6]
    end

    test "raises for DMatrix given invalid shape" do
      assert_raise(AnnexError, fn -> DMatrix.cast(@dmatrix_2_by_3, [7]) end)
    end
  end

  describe "is_type?/1" do
    test "true for DMatrix struct" do
      assert %DMatrix{} = @dmatrix_2_by_3
      assert DMatrix.is_type?(@dmatrix_2_by_3) == true
    end

    test "false for non-DMatrix struct" do
      assert DMatrix.is_type?(%URI{}) == false
    end
  end

  describe "to_flat_list/1" do
    test "works for DMatrix struct" do
      assert DMatrix.to_flat_list(@dmatrix_2_by_3) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    end
  end

  describe "shape/1" do
    test "returns 2D shape" do
      assert DMatrix.shape(@dmatrix_2_by_3) == [2, 3]
    end
  end

  describe "apply_op/3" do
    test "works for dot of DMatrixs" do
      d1 = DMatrix.build([[1.0, 1.0, 1.0]])

      d2 =
        DMatrix.build([
          [2.0, 3.0],
          [2.0, 3.0],
          [2.0, 3.0]
        ])

      assert DMatrix.apply_op(d1, :dot, [d2]) == DMatrix.build([[6.0, 9.0]])
    end

    test "works for add with number" do
      d1 = DMatrix.build([[1.0, 1.0, 1.0]])
      output = DMatrix.build([[3.0, 3.0, 3.0]])

      tensor =
        output
        |> DMatrix.tensor()
        |> Map.put(:identity, 2.0)

      assert DMatrix.apply_op(d1, :add, [2.0]) == %DMatrix{output | tensor: tensor}
    end

    test "works for add with another matrix" do
      d1 = DMatrix.build([[1.0, 1.0, 1.0]])
      assert DMatrix.apply_op(d1, :add, [d1]) == DMatrix.build([[2.0, 2.0, 2.0]])
    end

    test "works for subtract with number" do
      d1 = DMatrix.build([[1.0, 1.0, 1.0]])
      output = DMatrix.build([[-1.0, -1.0, -1.0]])

      tensor =
        output
        |> DMatrix.tensor()
        |> Map.put(:identity, -2.0)

      assert DMatrix.apply_op(d1, :subtract, [2.0]) == %DMatrix{output | tensor: tensor}
    end

    test "works for subtract with other matrix" do
      d1 = DMatrix.build([[1.0, 1.0, 1.0]])
      assert DMatrix.apply_op(d1, :subtract, [d1]) == DMatrix.build([[0.0, 0.0, 0.0]])
    end

    test "works for multiply with a number" do
      d1 = DMatrix.build([[1.0, 2.0, 3.0]])
      assert DMatrix.apply_op(d1, :multiply, [2.0]) == DMatrix.build([[2.0, 4.0, 6.0]])
    end

    test "works for multiply with another matrix" do
      d1 = DMatrix.build([[1.0, 2.0, 3.0]])
      assert DMatrix.apply_op(d1, :multiply, [d1]) == DMatrix.build([[1.0, 4.0, 9.0]])
    end

    test "works for transpose with no args" do
      d1 = DMatrix.build([[1.0, 2.0, 3.0]])
      assert DMatrix.shape(d1) == [1, 3]
      d2 = DMatrix.apply_op(d1, :transpose, [])
      assert DMatrix.shape(d2) == [3, 1]
      assert d2 == DMatrix.build([[1.0], [2.0], [3.0]])
    end

    test "maps when arg2 is an arity 1 func" do
      mapper = fn n -> n * n * n end
      d1 = DMatrix.build([[1.0, 2.0, 3.0]])
      assert DMatrix.apply_op(d1, :anything, [mapper]) == DMatrix.build([[1.0, 8.0, 27.0]])
    end
  end

  describe "new_random/2" do
    test "returns a DMatrix of the correct shape" do
      assert %DMatrix{} = dmatrix = DMatrix.new_random(2, 3)
      assert DMatrix.shape(dmatrix) == [2, 3]
    end

    test "returns a DMatrix full of random weights" do
      assert %DMatrix{} = dmatrix = DMatrix.new_random(2, 3)
      flat = DMatrix.to_flat_list(dmatrix)
      assert Enum.all?(flat, &is_float/1)
      assert Enum.all?(flat, fn x -> x >= -1.0 && x <= 1.0 end)
    end
  end

  describe "build/1" do
    test "returns a DMatrix with 1 row and n columns given flat data of n length" do
      assert [3.0, 4.0, 5.0]
             |> DMatrix.build()
             |> DMatrix.shape() == [1, 3]
    end

    test "returns a DMatrix with matching dimensions given a 2D list" do
      data_2d = [
        [1.0, 1.5, 2.0],
        [3.0, 4.0, 5.0]
      ]

      dmatrix = DMatrix.build(data_2d)

      assert DMatrix.shape(dmatrix) == [2, 3]
      assert DMatrix.to_flat_list(dmatrix) == [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    end
  end

  describe "build/3" do
    test "given flat data, rows, and columns returns a DMatrix struct for matching sizes" do
      floats = [1.0, 2.0, 3.0, 4.0, 5.0]
      assert %DMatrix{} = dmatrix = DMatrix.build(floats, 5, 1)
      assert DMatrix.shape(dmatrix) == [5, 1]
    end

    test "given flat data, rows, and columns raise for mismatching sizes" do
      floats = [1.0, 2.0, 3.0, 4.0, 5.0]
      assert_raise(AnnexError, fn -> DMatrix.build(floats, 6, 1) end)
    end
  end

  describe "ones/2" do
    test "returns a DMatrix with the correct shape" do
      assert %DMatrix{} = dmatrix = DMatrix.ones(6, 6)
      assert DMatrix.shape(dmatrix) == [6, 6]
    end

    test "returns all 1.0s" do
      DMatrix.ones(6, 6)
      |> DMatrix.to_flat_list()
      |> Enum.each(fn f ->
        assert f === 1.0
      end)
    end
  end

  describe "zeros/2" do
    test "returns a DMatrix with the correct shape" do
      assert %DMatrix{} = dmatrix = DMatrix.zeros(6, 6)
      assert DMatrix.shape(dmatrix) == [6, 6]
    end

    test "returns all 0.0s" do
      DMatrix.zeros(6, 6)
      |> DMatrix.to_flat_list()
      |> Enum.each(fn f ->
        assert f === 0.0
      end)
    end
  end

  describe "tensor/1" do
    test "returns the internal tensor of the DMatrix" do
      assert %DMatrix{} = @dmatrix_2_by_3
      assert %Tensor.Tensor{} = DMatrix.tensor(@dmatrix_2_by_3)
    end
  end

  describe "to_list_of_lists" do
    test "returns a matching 2D list" do
      assert DMatrix.to_list_of_lists(@dmatrix_2_by_3) == [
               [1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0]
             ]
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
      assert DMatrix.shape(d1) == [1, 3]
      assert DMatrix.shape(d2) == [3, 2]
      assert DMatrix.shape(d3) == [1, 2]
    end

    test "produces the correct value", %{d1: d1, d2: d2} do
      assert DMatrix.dot(d1, d2) == DMatrix.build([[6.0, 9.0]])
    end
  end

  describe "multiply/2" do
    test "element-wise multiplies a DMatrix by a number" do
      d1 = DMatrix.build([[1.0, 2.0, 3.0]])
      assert DMatrix.multiply(d1, 2.0) == DMatrix.build([[2.0, 4.0, 6.0]])
    end

    test "element-wise multiplies a DMatrix by another DMatrix" do
      d1 = DMatrix.build([[1.0, 2.0, 3.0]])
      assert DMatrix.multiply(d1, d1) == DMatrix.build([[1.0, 4.0, 9.0]])
    end

    test "raises for invalid shapes" do
      d1 = DMatrix.build([[1.0, 2.0, 3.0]])
      d2 = DMatrix.build([[1.0, 2.0, 3.0, 4.0]])

      assert_raise(AnnexError, fn ->
        DMatrix.multiply(d1, d2)
      end)
    end
  end

  describe "add/2" do
    test "adds a number to a DMatrix" do
      d1 = DMatrix.build([[1.0, 1.0, 1.0]])
      output = DMatrix.build([[3.0, 3.0, 3.0]])

      tensor =
        output
        |> DMatrix.tensor()
        |> Map.put(:identity, 2.0)

      assert DMatrix.add(d1, 2.0) == %DMatrix{output | tensor: tensor}
    end

    test "adds DMatrix and another DMatrix" do
      d1 = DMatrix.build([[1.0, 1.0, 1.0]])
      assert DMatrix.add(d1, d1) == DMatrix.build([[2.0, 2.0, 2.0]])
    end

    test "raises for invalid shapes" do
      d1 = DMatrix.build([[1.0, 1.0, 1.0]])
      d2 = DMatrix.build([[1.0, 1.0, 1.0, 1.0]])

      assert_raise(AnnexError, fn ->
        DMatrix.add(d1, d2)
      end)
    end
  end

  describe "subtract/2" do
    test "subtracts a number from a DMatrix" do
      d1 = DMatrix.build([[1.0, 1.0, 1.0]])
      output = DMatrix.build([[-1.0, -1.0, -1.0]])

      tensor =
        output
        |> DMatrix.tensor()
        |> Map.put(:identity, -2.0)

      assert DMatrix.subtract(d1, 2.0) == %DMatrix{output | tensor: tensor}
    end

    test "subtracts DMatrix and another DMatrix" do
      d1 = DMatrix.build([[1.0, 1.0, 1.0]])
      assert DMatrix.subtract(d1, d1) == DMatrix.build([[0.0, 0.0, 0.0]])
    end

    test "raises for invalid shapes" do
      d1 = DMatrix.build([[1.0, 1.0, 1.0]])
      d2 = DMatrix.build([[1.0, 1.0, 1.0, 1.0]])

      assert_raise(AnnexError, fn ->
        DMatrix.subtract(d1, d2)
      end)
    end
  end

  describe "transpose/1" do
    test "transposes the DMatrix" do
      transposed = DMatrix.transpose(@dmatrix_2_by_3)
      assert DMatrix.shape(@dmatrix_2_by_3) == [2, 3]
      assert DMatrix.shape(transposed) == [3, 2]
      assert DMatrix.to_flat_list(@dmatrix_2_by_3) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      assert DMatrix.to_flat_list(transposed) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    end
  end

  describe "map/2" do
    test "maps the values of a DMatrix" do
      assert DMatrix.to_flat_list(@dmatrix_2_by_3) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

      assert DMatrix.map(@dmatrix_2_by_3, fn x -> x * x * x end) ==
               DMatrix.build([
                 [1.0, 8.0, 27.0],
                 [64.0, 125.0, 216.0]
               ])
    end
  end

  describe "Enumerable protocol" do
    test "Enum.count/1 works" do
      assert Enum.count(@dmatrix_2_by_3) == 6
    end

    test "Enum.member?/2 works" do
      assert Enum.member?(@dmatrix_2_by_3, 1.0)
    end
  end
end
