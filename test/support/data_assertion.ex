defmodule Annex.DataAssertion do
  @moduledoc """
  A helper module for making assertions about the returns of a Data
  type's callbacks.
  """
  use ExUnit.CaseTemplate

  alias Annex.{
    Data,
    Data.Shape
  }

  def cast(type, data, :defer) do
    casted = Data.cast(type, data, :defer)

    assert casted == data, """
    Annex.Data.cast/3 failed to produce the exact same data when given :defer as the shape

    invalid_result: #{inspect(casted)}

    type: #{inspect(type)}
    data: #{inspect(data)}
    shape: :defer
    """

    casted
  end

  def cast(type, data, shape) when is_tuple(shape) do
    product = Shape.product(shape)

    flat_data =
      if is_type?(type, data) do
        to_flat_list(type, data)
      else
        Enum.into(data, [])
      end

    n_elements = length(flat_data)

    assert product == n_elements, """
    The number of elements in a flattened data structure must be the same as the product of the
    elements of a shape.

    product: #{inspect(product)}
    n_elements: #{inspect(n_elements)}

    type: #{inspect(type)}
    data: #{inspect(data)}
    shape: #{inspect(shape)}
    """

    casted = Data.cast(type, data, shape)

    assert Data.is_type?(type, casted) == true, """
    Data.cast/3 failed to produce the expected type.

    invalid_result: #{inspect(casted)}

    type: #{inspect(type)}
    data: #{inspect(data)}
    shape: #{inspect(shape)}

    """

    casted
  end

  def to_flat_list(type, data) do
    flattened = Data.to_flat_list(type, data)

    assert is_list(flattened) == true

    assert Enum.all?(flattened, &is_float/1), """
      Data.to_flat_list/2 failed to produce a flat list of floats.

      invalid_result: #{inspect(flattened)}

      type: #{inspect(type)}
      data: #{inspect(data)}

    """

    flattened
  end

  def is_type?(type, data) do
    result = Data.is_type?(type, data)

    assert result in [true, false], """
    Annex.Data.is_type?/2 failed to return a boolean.

    invalid_result: #{inspect(result)}

    type: #{inspect(type)}
    data: #{inspect(data)}

    """

    result
  end

  def shape(type, data) do
    result = Data.shape(type, data)

    assert Shape.is_shape?(result) == true, """
    Annex.Data.shape/2 failed to return a valid shape.

    invalid_result: #{inspect(result)}

    type: #{inspect(type)}
    data: #{inspect(data)}

    """

    result
  end

  def shape_is_valid(type, data) do
    shape = Data.shape(type, data)

    assert Shape.is_shape?(shape), """
    For Annex.Data.shape/2 failed to produce a valid shape.

    A shape must be a tuple of integers (non-empty) or the atom :defer.

    invalid_shape: #{inspect(shape)}

    type: #{inspect(type)}
    data: #{inspect(data)}
    """

    shape_tuple_is_all_integers(shape)
  end

  def shape_tuple_is_all_integers(shape) do
    all_ints? =
      shape
      |> Tuple.to_list()
      |> Enum.all?(&is_integer/1)

    assert all_ints? == true, """
    Annex.Data.shape/2 failed to produce a valid shape.

    A tuple shape have integer elements only.

    invalid_shape: #{inspect(shape)}
    """

    all_ints?
  end

  def shape_product(:defer) do
    raise """
    Shape.product/1 was attempted to be called during testing.
    """
  end

  def shape_product(shape) when is_tuple(shape) do
    assert shape_tuple_is_all_integers(shape) == true
    product = Shape.product(shape)

    assert is_integer(product) == true, """
    Shape.product/1 failed to produce an integer.

    invalid_result: #{inspect(product)}

    shape: #{inspect(shape)}
    """

    product
  end
end
