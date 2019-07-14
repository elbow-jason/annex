defmodule Annex.DataCase do
  @moduledoc """
  Annex.DataCase is an ExUnit.CaseTemplate with helpers to validing the implementation of
  Annex.Data behaviours.
  """

  use ExUnit.CaseTemplate

  alias Annex.{
    Data.List1D,
    DataAssertion
  }

  using kwargs do
    quote do
      @data_type Keyword.fetch!(unquote(kwargs), :type)
      @datas_and_shapes Keyword.fetch!(unquote(kwargs), :data)
      @pretty inspect(@data_type)
      alias Annex.DataCase

      test "Annex.DataCase: #{@pretty} to_flat_list/1 callback works" do
        DataCase.test_to_flat_list(@data_type, @datas_and_shapes)
      end

      test "Annex.DataCase: #{@pretty} shape/1 callback works" do
        DataCase.test_shape(@data_type, @datas_and_shapes)
      end

      test "Annex.DataCase: #{@pretty} cast/2 callback works" do
        DataCase.test_cast(@data_type, @datas_and_shapes)
      end

      test "Annex.DataCase: #{@pretty} full conversion works" do
        DataCase.test_conversion(@data_type, @datas_and_shapes)
      end
    end
  end

  def run_all_assertions(type, datas_and_shapes) do
    test_to_flat_list(type, datas_and_shapes)
    test_shape(type, datas_and_shapes)
    test_cast(type, datas_and_shapes)
    test_conversion(type, datas_and_shapes)
  end

  @doc """
  Tests that a type has can be correctly converted

  This macro relies on the correct implementation of Annex.Data.List1D.
  """
  def test_conversion(_type, []) do
    []
  end

  def test_conversion(type, [first | rest]) do
    [test_conversion(type, first) | test_conversion(type, rest)]
  end

  def test_conversion(type, {data, _expected_shape, _target_shape}) do
    # get the data's shape
    # get the flat data
    # get the flat list's shape
    DataAssertion.shape_is_valid(type, data)
    shape = DataAssertion.shape(type, data)

    flat_data = DataAssertion.to_flat_list(type, data)
    list_shape = DataAssertion.shape(List1D, flat_data)

    # the tested data type should be the same as the list type so conversion can
    # happen back and forth
    data_shape_product = DataAssertion.shape_product(shape)
    list_shape_product = DataAssertion.shape_product(list_shape)

    assert data_shape_product == list_shape_product, """
    The shape for #{inspect(type)} did not match the shape for the List1D.

    data_shape_product: #{inspect(data_shape_product)}
    list_shape_product: #{inspect(list_shape_product)}

    list_shape: #{inspect(list_shape)}
    data_shape: #{inspect(shape)}

    data: #{inspect(data)}
    flat_data: #{inspect(flat_data)}
    """

    # casting the list_flat_data and list_shape into the give type should end up with
    # the same shapes and data again.
    # exact comparision of the given data and the casted data cannot be relied upon
    # due to the unknown nature of the underlying data structure.
    casted = DataAssertion.cast(type, flat_data, shape)
    assert DataAssertion.shape(type, data) == DataAssertion.shape(type, casted)
    assert DataAssertion.to_flat_list(type, data) == DataAssertion.to_flat_list(type, casted)
  end

  @doc """
  Tests the implemenation of to_flat_list for a type.
  """
  def test_to_flat_list(_type, []) do
    []
  end

  def test_to_flat_list(type, [first | rest]) do
    [test_to_flat_list(type, first) | test_to_flat_list(type, rest)]
  end

  def test_to_flat_list(type, {data, _shape, _target_shape}) do
    DataAssertion.to_flat_list(type, data)
  end

  @doc """
  Tests the implementation of a type's cast/3 function.
  """
  def test_cast(_type, []) do
    []
  end

  def test_cast(type, [first | rest]) do
    [test_cast(type, first) | test_cast(type, rest)]
  end

  def test_cast(type, {data, _expected_shape, target_shape}) do
    DataAssertion.cast(type, data, target_shape)
  end

  @doc """
  Tests the implementation of a type's shape/1 function.
  """
  def test_shape(_type, []) do
    []
  end

  def test_shape(type, [first | rest]) do
    [test_shape(type, first) | test_shape(type, rest)]
  end

  def test_shape(type, {data, expected_shape, _target_shape}) do
    assert DataAssertion.shape_is_valid(type, data) == true
    result = DataAssertion.shape(type, data)

    assert result == expected_shape, """
    #{inspect(type)}.shape/1 failed to produce the expected shape.

    expected_shape: #{inspect(expected_shape)}
    invalid_result: #{inspect(result)}
    """
  end

  # defp assert_many_conversion(type, datas_and_shapes) do
  #   Enum.each(datas_and_shapes, fn {data, _shape, _target_shape} ->
  #     assert_one_conversion(type, data)
  #   end)
  # end

  # defp assert_one_conversion(type, data) do
  # end
end
