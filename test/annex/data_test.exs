defmodule Annex.DataTest do
  use Annex.DataCase
  alias Annex.Data
  alias AnnexHelpers.SimpleData

  @simple_3 %SimpleData{data: [1.0, 2.0, 3.0], shape: {3}}

  @simple_4_by_5 %SimpleData{
    data: [
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0],
      [17.0, 18.0, 19.0, 20.0]
    ],
    shape: {4, 5}
  }

  test_cast(SimpleData, %SimpleData{})
  test_to_flat_list(SimpleData, [@simple_3, @simple_4_by_5])
  test_conversion(SimpleData, [@simple_3, @simple_4_by_5])

  describe "cast/3" do
    test "calls cast for implementers of behaviour" do
      data = [1.0, 2.0, 3.0]
      shape = {3, 1}

      assert Data.cast(SimpleData, data, shape) == %SimpleData{
               data: data,
               shape: shape
             }
    end
  end

  describe "to_flat_list/2" do
    test "works" do
      simple = %SimpleData{
        data: [3.0, 2.0, 1.0],
        shape: {3}
      }

      assert Data.to_flat_list(SimpleData, simple) == [3.0, 2.0, 1.0]
    end

    test "errors for non-implementers of Enumerable" do
      simple = %SimpleData{
        data: [3.0, 2.0, 1.0]
      }

      assert Data.to_flat_list(SimpleData, simple) == [3.0, 2.0, 1.0]
    end
  end
end
