defmodule Annex.Data do
  @type data :: any()
  @type shape :: tuple() | :any

  @callback cast(data, shape) :: data()
  @callback to_flat_list(data) :: list(float())
  @callback shape(data) :: shape()

  @doc """
  Annex.Data.cast/4 calls cast/3 for an Annex.Data behaviour implementing module.

  Valid shapes are a non-empty tuple of positive integers or the atom `:any`
  """
  def cast(type, data, {}) when is_list(data) do
    raise ArgumentError,
      message: """
      Annex.Data.cast got an empty tuple for shape.

      Suggestion: use `:any` to indicate a lack of shape, an empty tuple is an invalid shape.

      type: #{inspect(type)}
      data: #{inspect(data)}
      """
  end

  def cast(type, data, shape) when is_tuple(shape) and is_atom(type) do
    type.cast(data, shape)
  end

  def cast(type, data, :any) when is_atom(type) do
    type.cast(data, :any)
  end

  @doc """
  Flattens an Annex.Data into a list of floats.
  """
  def to_flat_list(type, data), do: type.to_flat_list(data)

  @doc """
  Given an Annex.Data `type` and the `data` returns the shape of the data.

  The shape of data is used to cast between the expected shapes from one Annex.Layer
  to the next or from one Annex.Sequence to the next.
  """
  def shape(type, data), do: type.shape(data)
end
