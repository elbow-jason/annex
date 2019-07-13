defmodule Annex.DataCase do
  use ExUnit.CaseTemplate

  alias Annex.{
    Data,
    Data.ListType
  }

  using do
    quote do
      use ExUnit.Case
      import Annex.DataCase
    end
  end

  @doc """
  Tests that a type has can be correctly converted

  This macro relies on the correct implementation of Data.ListType.
  """
  defmacro test_conversion(type, list_of_datas) when is_list(list_of_datas) do
    quote do
      test "conversion of #{inspect(unquote(type))} works" do
        Annex.DataCase.assert_many_conversion(unquote(type), unquote(list_of_datas))
      end
    end
  end

  @doc """
  Tests the implemenation of to_flat_list for a type.
  """
  defmacro test_to_flat_list(type, list_of_data) do
    quote do
      test "to_flat_list of #{inspect(unquote(type))} works" do
        Annex.DataCase.assert_many_to_flat_list(unquote(type), unquote(list_of_data))
      end
    end
  end

  def assert_many_to_flat_list(type, list_of_data) do
    Enum.each(list_of_data, fn data ->
      assert_one_to_flat_list(type, data)
    end)
  end

  def assert_one_to_flat_list(type, data) do
    flattened = Data.to_flat_list(type, data)

    assert is_list(flattened)

    assert Enum.all?(flattened, &is_float/1), """
      Expected Data.to_flat_list/2  to return a flat list of floats.
      type: #{inspect(type)}
      data: #{inspect(data)}
    """
  end

  def assert_shape_is_valid(type, data) do
    shape = Data.shape(type, data)

    assert shape == :any or is_tuple(shape), """
    For Annex.Data a shape must be a tuple of integers or the atom :any.

    invalid_shape: #{inspect(shape)}
    type: #{inspect(type)}
    data: #{inspect(data)}
    """

    if is_tuple(shape) do
      assert tuple_size(shape) > 0, """
      For Annex.Data a shape tuple cannot be empty.

      invalid_shape: #{inspect(shape)}
      type: #{inspect(type)}
      data: #{inspect(data)}
      """

      assert shape |> Tuple.to_list() |> Enum.all?(&is_integer/1), """
      For Annex.Data a tuple shape have integer elements only.

      invalid_shape: #{inspect(shape)}
      type: #{inspect(type)}
      data: #{inspect(data)}
      """
    end
  end

  def assert_many_conversion(type, list_of_datas) do
    Enum.each(list_of_datas, fn data -> assert_one_conversion(type, data) end)
  end

  def assert_one_conversion(type, data) do
    # get the data's shape
    assert_shape_is_valid(type, data)

    shape = Data.shape(type, data)
    # make sure to_flat_list works
    assert_one_to_flat_list(type, data)
    # get the flat data
    flat_data = Data.to_flat_list(type, data)

    # turn the data into a list representation
    list_data = Data.cast(ListType, flat_data, shape)
    # get the list's shape
    list_shape = Data.shape(ListType, list_data)

    # the tested data type should be the same as the list type so conversion can
    # happen back and forth
    assert shape == list_shape, """
    The shape for #{inspect(type)} did not match the shape for the ListType.

    expected_shape: #{inspect(list_shape)}
    got_shape: #{inspect(shape)}
    data: #{inspect(data)}
    list_data: #{inspect(list_data)}
    """

    list_flat_data = Data.to_flat_list(ListType, list_data)
    assert flat_data == list_flat_data

    # casting the list_flat_data and list_shape into the give type should end up with
    # the same shapes and data again.
    # exact comparision of the given data and the casted data cannot be relied upon
    # due to the unknown nature of the underlying data structure.
    casted = Data.cast(type, list_flat_data, list_shape)
    assert Data.shape(type, data) == Data.shape(type, casted)
    assert Data.to_flat_list(type, data) == Data.to_flat_list(type, casted)
  end

  defmacro test_cast(type, basic_pattern, datas_and_shapes \\ [])
           when is_list(datas_and_shapes) do
    quote do
      flat_data = {[1.0, 2.0, 3.0], {3}}
      any_data = {[2.0, 3.0, 4.0], :any}
      all_data = [flat_data, any_data | unquote(datas_and_shapes)]

      Enum.each(all_data, fn {data, shape} ->
        result = Data.cast(unquote(type), data, shape)
        assert match?(unquote(basic_pattern), result)
        result
      end)
    end
  end
end
