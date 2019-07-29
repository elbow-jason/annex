# defmodule Annex.Data.TensorTest do
#   use ExUnit.Case
#   alias Annex.Data.DTensor

#   @data_10 [
#     1.0,
#     2.0,
#     3.0,
#     4.0,
#     5.0,
#     6.0,
#     7.0,
#     8.0,
#     9.0,
#     10.0
#   ]
#   setup do
#     d_2_by_5 = %DTensor{
#       data: @data_10,
#       shape: {2, 5}
#     }

#     {:ok, d_2_by_5: d_2_by_5}
#   end

#   describe "build/2" do
#     test "works for flat list of floats" do
#       shape = {2, 5}

#       assert DTensor.build(@data_10, shape) == %DTensor{
#                data: @data_10,
#                shape: shape
#              }
#     end

#     test "works for generators" do
#       ones_func = fn -> 1.0 end

#       expected_data =
#         ones_func
#         |> Stream.repeatedly()
#         |> Enum.take(10)

#       shape = {2, 5}

#       assert DTensor.build(ones_func, shape) == %DTensor{
#                data: expected_data,
#                shape: shape
#              }
#     end

#     test "when debug is true a data vs shape mismatch is raised" do
#       data =
#         1.0
#         |> Stream.iterate(fn x -> x + 1.0 end)
#         |> Enum.take(10)

#       shape = {3, 5}

#       err = assert_raise(Annex.AnnexError, fn -> DTensor.build(data, shape) end)
#       assert err.message =~ "data size must match shape size"
#     end

#     test "when debug is true a data with a non-float raises" do
#       assert DTensor.__annex__(:debug) == true
#       data = [:nope, 2.0, 3.0, 4.0]
#       shape = {2, 2}

#       err = assert_raise(Annex.AnnexError, fn -> DTensor.build(data, shape) end)
#       assert err.message =~ "data must be a flat list of floats"
#     end
#   end

#   describe "shape/1" do
#     test "works", %{d_2_by_5: matrix} do
#       assert DTensor.shape(matrix) == {2, 5}
#     end
#   end

#   describe "data/1" do
#     test "works", %{d_2_by_5: matrix} do
#       assert DTensor.data(matrix) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
#     end
#   end

#   describe "size/1" do
#     test "works", %{d_2_by_5: matrix} do
#       assert DTensor.size(matrix) == 10
#     end
#   end

#   describe "ones/1" do
#     test "works" do
#       d_tensor = DTensor.ones({3, 4})
#       assert DTensor.shape(d_tensor) == {3, 4}
#       data = DTensor.data(d_tensor)

#       assert length(data) == 12
#       assert Enum.all?(data, fn w -> w == 1.0 end)
#     end
#   end

#   describe "zeros/1" do
#     test "works" do
#       d_tensor = DTensor.zeros({3, 4})
#       assert DTensor.shape(d_tensor) == {3, 4}
#       data = DTensor.data(d_tensor)

#       assert length(data) == 12
#       assert Enum.all?(data, fn w -> w == 0.0 end)
#     end
#   end

#   describe "to_list/1" do
#     test "works", %{d_2_by_5: tensor} do
#       assert DTensor.to_list(tensor) == [
#                [1.0, 2.0, 3.0, 4.0, 5.0],
#                [6.0, 7.0, 8.0, 9.0, 10.0]
#              ]
#     end
#   end

#   describe "to_flat_list/1" do
#     test "works", %{d_2_by_5: tensor} do
#       assert DTensor.to_flat_list(tensor) == @data_10
#     end
#   end

#   describe "transpose/1" do
#     test "returns the exact same DTensor for 1 dimension" do
#       data = [1.0, 2.0, 3.0, 4.0]

#       data_t =
#         data
#         |> DTensor.build({4})
#         |> DTensor.transpose()
#         |> DTensor.to_list()

#       assert data_t == data
#     end

#     test "works for 2 dimensional tensor" do
#       data = [1.0, 2.0, 3.0, 4.0]

#       data_t =
#         data
#         |> DTensor.build({2, 2})
#         |> DTensor.transpose()
#         |> DTensor.to_list()

#       assert data_t == [
#                [1.0, 3.0],
#                [2.0, 4.0]
#              ]
#     end

#     test "works for non-square 2 dimensional tensor" do
#       raw_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
#       data = DTensor.build(raw_data, {2, 5})

#       assert DTensor.to_list(data) == [
#                [1.0, 2.0, 3.0, 4.0, 5.0],
#                [6.0, 7.0, 8.0, 9.0, 10.0]
#              ]

#       data_t =
#         data
#         |> DTensor.transpose()
#         |> DTensor.to_list()

#       # YAY
#       assert data_t == [
#                [1.0, 6.0],
#                [2.0, 7.0],
#                [3.0, 8.0],
#                [4.0, 9.0],
#                [5.0, 10.0]
#              ]
#     end

#     test "raises for 3 or more dimensional tensor" do
#       raw_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
#       data = DTensor.build(raw_data, {2, 2, 2})
#       assert DTensor.__annex__(:debug) == true
#       assert_raise(Annex.AnnexError, fn -> DTensor.transpose(data) end)
#     end
#   end

#   describe "dot/2" do
#     test "works" do
#       left = DTensor.build([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], {5, 2})
#       right = DTensor.build([2.0, 4.0], {2, 1})

#       expected_result = [
#         [110.0],
#         [260.0]
#       ]

#       assert DTensor.dot(left, right) == expected_result
#     end
#   end
# end
