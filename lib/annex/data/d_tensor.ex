# defmodule Annex.Data.DTensor do
#   @moduledoc """
#   A matrix library that does not do strange things.
#   """
#   alias Annex.{
#     Data.Shape,
#     Data.DTensor,
#     # DataError,
#     Utils
#   }

#   use Annex.Debug, debug: true

#   @enforce_keys [:data, :shape]

#   defstruct [:data, :shape]

#   def build(data, shape) when is_list(data) do
#     debug_assert "data size must match shape size" do
#       data_size = length(data)
#       shape_size = Shape.product(shape)
#       data_size == shape_size
#     end

#     debug_assert "data must be a flat list of floats" do
#       Enum.all?(data, &is_float/1)
#     end

#     %DTensor{
#       data: data,
#       shape: shape
#     }
#   end

#   def build(generator, shape) when is_function(generator, 0) do
#     generator
#     |> Stream.repeatedly()
#     |> Enum.take(Shape.product(shape))
#     |> build(shape)
#   end

#   def shape(%DTensor{shape: shape}), do: shape

#   def data(%DTensor{data: data}), do: data

#   def size(%DTensor{} = d) do
#     d
#     |> shape()
#     |> Shape.product()
#   end

#   def n_dimensions(%DTensor{} = d) do
#     d
#     |> shape()
#     |> tuple_size()
#   end

#   def ones(shape) when is_tuple(shape) do
#     build(fn -> 1.0 end, shape)
#   end

#   def zeros(shape) when is_tuple(shape) do
#     build(fn -> 0.0 end, shape)
#   end

#   def to_list(%DTensor{} = d_tensor) do
#     d_tensor
#     |> shape()
#     |> Tuple.to_list()
#     |> Enum.reverse()
#     |> Enum.reduce(to_flat_list(d_tensor), fn chunk_size, acc ->
#       Enum.chunk_every(acc, chunk_size)
#     end)
#     |> unwrap()
#   end

#   def to_flat_list(%DTensor{} = d_tensor) do
#     data(d_tensor)
#   end

#   def transpose(%DTensor{} = d_tensor) do
#     shape = shape(d_tensor)

#     debug_assert "DTensor.transpose/1 is only implemented for shapes that are 1 or 2 dimensions" do
#       n_dimensions = tuple_size(shape)
#       n_dimensions == 1 or n_dimensions == 2
#     end

#     case shape do
#       {_} ->
#         d_tensor

#       {rows, columns} ->
#         rev_data =
#           d_tensor
#           |> data()
#           |> Enum.reverse()

#         fn -> [] end
#         |> Stream.repeatedly()
#         |> Enum.take(columns)
#         |> do_transpose(rev_data, shape)
#         |> build({columns, rows})
#     end
#   end

#   defp do_transpose(acc, [], {_, _}) do
#     acc
#     |> Enum.reverse()
#     |> List.flatten()
#   end

#   defp do_transpose(acc, data, {_rows, columns} = shape) do
#     {column_data, rest_data} = Enum.split(data, columns)

#     column_data
#     |> Enum.zip(acc)
#     |> Enum.map(fn {datum, row_acc} -> [datum | row_acc] end)
#     |> do_transpose(rest_data, shape)
#   end

#   def dot(%DTensor{} = left, %DTensor{} = right) do
#     debug_assert "left must be a 2D tensor" do
#       left_shape = shape(left)
#       tuple_size(left_shape) == 2
#     end

#     debug_assert "right must be a 2D tensor" do
#       right_shape = shape(right)
#       tuple_size(right_shape) == 2
#     end

#     debug_assert "left_columns must the the same as right rows" do
#       {_, left_columns} = shape(left)
#       {right_rows, _} = shape(right)

#       left_columns == right_rows
#     end

#     right_list = to_list(right)

#     left
#     |> DTensor.to_list()
#     |> Enum.map(fn left_row ->
#       Enum.map(right_list, fn right_row ->
#         IO.inspect({right_row, left_row}, label: :laft_N_rigt)

#         left_row
#         |> Utils.zipmap(right_row, fn lx, rx -> lx * rx end)
#         |> Enum.sum()
#       end)
#     end)

#     # list_of_lists =
#     #   for r <- 0..(m - 1) do
#     #     for c <- 0..(p - 1) do
#     #       Vector.dot_product(a[r], b_t[c])
#     #     end
#     #   end

#     # Tensor.new(list_of_lists, [m, p])
#   end

#   def map(%DTensor{} = d_tensor, fun) do
#     %DTensor{d_tensor | data: d_tensor |> data() |> Enum.map(fun)}
#   end

#   defp unwrap([unwrapped]), do: unwrapped
# end
