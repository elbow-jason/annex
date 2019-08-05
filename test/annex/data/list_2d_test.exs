defmodule Annex.Data.List2DTest do
  alias Annex.{
    Data,
    Data.List2D
  }

  require List2D

  @data_2_by_3 [
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0]
  ]

  @data_4_by_2 [
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0]
  ]

  @casts [
    {@data_2_by_3, [2, 3], [6, 1]},
    {@data_2_by_3, [2, 3], [3, 2]},
    {@data_2_by_3, [2, 3], [2, 3]},
    {@data_2_by_3, [2, 3], [1, 6]},
    {@data_4_by_2, [4, 2], [1, 8]},
    {@data_4_by_2, [4, 2], [2, 4]},
    {@data_4_by_2, [4, 2], [4, 2]},
    {@data_4_by_2, [4, 2], [8, 1]}
  ]

  use Annex.DataCase, type: List2D, data: @casts

  describe "cast/2" do
    test "given flat data paired with a valid shape can convert into a 2D List" do
      assert List2D.cast([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]) == [
               [1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0]
             ]
    end

    test "given a 2D list of floats and the same shape returns the same 2D list" do
      assert Data.shape(List2D, @data_2_by_3) == [2, 3]
      assert List2D.cast(@data_2_by_3, [2, 3]) == @data_2_by_3
    end

    test "works for 2D where the dimensions is the count of the data items" do
      assert List2D.cast(@data_2_by_3, [3, 2]) == [[1.0, 1.0], [1.0, 2.0], [2.0, 2.0]]
    end

    test "raises for other than 2D list of floats" do
      raises = fn data ->
        assert_raise(ArgumentError, fn -> List2D.cast(data, [10, 10]) end)
      end

      raises.([[1]])
      raises.(:other)
      raises.("etc")
    end
  end

  describe "to_flat_list/1" do
    test "can handle nested lists" do
      assert List2D.to_flat_list(@data_2_by_3) == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
      assert List2D.to_flat_list(@data_4_by_2) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    end

    test "raises for other than 2D list of floats" do
      raises = fn data ->
        assert_raise(ArgumentError, fn -> List2D.to_flat_list(data) end)
      end

      raises.([[1]])
      raises.(:other)
      raises.("etc")
    end
  end

  describe "" do
    test "is correct for nested lists" do
      assert List2D.shape(@data_2_by_3) == [2, 3]
      assert List2D.shape(@data_4_by_2) == [4, 2]
    end

    test "raises for other than 2D list" do
      raises = fn data ->
        assert_raise(ArgumentError, fn -> List2D.shape(data) end)
      end

      raises.("bad thing")
      raises.('bad thing')
      raises.(:not_valid)
    end
  end

  describe "is_type?/1" do
    test "is true for float containing nested list" do
      assert List2D.is_type?(@data_2_by_3) == true
    end

    test "is false for other than 2D list" do
      assert List2D.is_type?([1.0, 2.0, 3.0]) == false
      assert List2D.is_type?(:nope) == false
      assert List2D.is_type?([:nope]) == false
      assert List2D.is_type?([1]) == false
      assert List2D.is_type?([[1]]) == false
      assert List2D.is_type?(1) == false
      assert List2D.is_type?(1.0) == false
      assert List2D.is_type?("1.0") == false
      assert List2D.is_type?('1.0') == false
      assert List2D.is_type?(true) == false
      assert List2D.is_type?(false) == false
      assert List2D.is_type?(%URI{}) == false
      assert List2D.is_type?(%{}) == false
    end
  end
end
