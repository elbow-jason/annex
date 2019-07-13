defmodule Annex.Data.ListType do
  @moduledoc """
  The Annex.Data.List is the most basic Annex.Data.
  """
  @behaviour Annex.Data

  def cast(data, :any) when is_list(data) do
    List.flatten(data)
  end

  def cast(data, shape) when is_list(data) do
    flat_data = List.flatten(data)
    dimensions = Tuple.to_list(shape)

    elements_count = length(flat_data)
    elements_expected = product(dimensions)

    if elements_count != elements_expected do
      raise ArgumentError,
        message: """
        The number of items in the provided data did not match the required number of items of the given
        shape.

        shape: #{inspect(shape)}
        expected_count: #{inspect(elements_count)}
        actual_count: #{inspect(elements_count)}
        data: #{inspect(data)}
        """
    end

    dimensions
    |> data_to_tensor(flat_data)
    |> unwrap()
  end

  def cast(other, shape) do
    other
    |> Enum.into([])
    |> cast(shape)
  end

  def to_flat_list([f | _] = data) when is_float(f), do: data
  def to_flat_list(data) when is_list(data), do: List.flatten(data)

  def shape([f | _] = data) when is_float(f), do: {length(data)}
  def shape(data) when is_list(data), do: do_shape(data, [])

  defp do_shape([first | _] = dim, acc) do
    do_shape(first, [length(dim) | acc])
  end

  defp do_shape(_, acc) do
    acc
    |> Enum.reverse()
    |> List.to_tuple()
  end

  defp product(dimensions) do
    Enum.reduce(dimensions, fn n, acc -> n * acc end)
  end

  defp data_to_tensor(dimensions, flat_data) do
    dimensions
    |> Enum.reverse()
    |> Enum.reduce(flat_data, fn chunk_size, acc ->
      Enum.chunk_every(acc, chunk_size)
    end)
  end

  defp unwrap([unwrapped]), do: unwrapped
end
