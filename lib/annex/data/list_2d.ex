defmodule Annex.Data.List2D do
  @moduledoc """
  List2D is a 2 dimensional list of lists of floats.
  """

  use Annex.Data

  @type t :: [[float(), ...], ...]

  @doc """
  Given flat `data` and a 2-D `shape` (in the form of `{rows, columns}`) returns
  a list of lists, a 2-D list.
  """
  @spec cast(Data.flat_data(), Shape.t()) :: t()
  def cast(data, {_rows, _columns} = shape) do
    flat_data =
      data
      |> type_check()
      |> List.flatten()

    elements_count = length(flat_data)

    elements_expected = Shape.product(shape)

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

    Data.flat_data_to_tensor(flat_data, shape)
  end

  @doc """
  Returns true for a list of lists of floats.
  """
  @spec is_type?(Data.data()) :: boolean
  def is_type?(data), do: is_list2D(data)

  @doc """
  The shape of a List2D can be calculated thus:

  `rows` is the number of elements in the outermost list.
  `columns` is the count of the elements of the first row.
  """
  def shape(data) do
    [row_of_floats | _] = type_check(data)
    {length(data), length(row_of_floats)}
  end

  @doc """
  The shape of a List2D can be calculated thus:

  `rows` is the number of elements in the outermost list.
  `columns` is the count of the elements of the first row.
  """
  def to_flat_list(data) do
    data
    |> type_check()
    |> List.flatten()
  end

  defp type_check(data) do
    if not is_list2D(data) do
      raise ArgumentError,
        message: """
        #{inspect(Annex.Data.List2D)} requires data to be a list of lists of floats.
        """
    end

    data
  end

  defp is_list2D([[f | _] | _]) when is_float(f), do: true
  defp is_list2D(_), do: false
end
