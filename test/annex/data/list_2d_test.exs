defmodule Annex.Data.List2DTest do
  use Annex.DataCase

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

  describe "cast/2" do
    test "given a 2D list of floats and the same shape returns the same 2D list" do
      assert Data.shape(List2D, @data_2_by_3) == {2, 3}
      assert List2D.cast(@data_2_by_3, {2, 3}) == @data_2_by_3
    end

    test "works for 2D where the dimensions is the count of the data items" do
      assert List2D.cast(@data_2_by_3, {3, 2}) == [[1.0, 1.0], [1.0, 2.0], [2.0, 2.0]]
    end

    test "raises for other than 2D list of floats" do
      raises = fn data ->
        assert_raise(ArgumentError, fn -> List2D.cast(data, {10, 10}) end)
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
    test "is correctly ordered for nested lists" do
      assert List2D.shape(@data_2_by_3) == {2, 3}
      assert List2D.shape(@data_4_by_2) == {4, 2}
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

  # describe "Data Behaviour" do
  #   test "shape/1 works" do
  #     [{@data_5, {5}}, {@data_2_by_3, {2, 3}}]
  #     |> Enum.each(fn {data, expected_shape} ->
  #       assert Data.shape(ListType, data) == expected_shape
  #     end)
  #   end

  #   test "cast/2 works" do
  #     [
  #       {@data_5, {5}, @data_5},
  #       {@data_2_by_3, {2, 3}, @data_2_by_3},
  #       {@data_8, {2, 2, 2}, @data_2_by_2_by_2},
  #       {@data_8, {4, 2}, @data_4_by_2}
  #     ]
  #     |> Enum.each(fn {data, shape_to_cast, expected} ->
  #       assert Data.cast(ListType, data, shape_to_cast) == expected
  #     end)
  #   end

  #   test "to_flat_list/1 works" do
  #     [
  #       {@data_5, @data_5},
  #       {@data_2_by_3, @data_6},
  #       {@data_6, @data_6},
  #       {@data_8, @data_8},
  #       {@data_2_by_2_by_2, @data_8},
  #       {@data_4_by_2, @data_8}
  #     ]
  #     |> Enum.each(fn {data, expected} ->
  #       assert Data.to_flat_list(ListType, data) == expected
  #     end)
  #   end
  # end
end
