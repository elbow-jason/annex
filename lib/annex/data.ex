defmodule Annex.Data do
  @moduledoc """
  Annex.Data defines the callbacks and helpers for data structures used
  by Annex.

  An implementer of the Annex.Layer behaviour must return an Annex.Data
  implementer from the `c:data_type/0` callback.
  """
  alias Annex.{
    Data,
    Data.Shape
  }

  @typedoc """
  A module that implements the Annex.Data Behaviour.
  """
  @type type :: module() | :defer
  @type data :: any()
  @type flat_data :: [float(), ...]

  defguard is_flat_data(data) when is_list(data) and is_float(hd(data))

  @callback cast(data, Shape.t()) :: data()
  @callback to_flat_list(data) :: list(float())
  @callback shape(data) :: Shape.t()
  @callback is_type?(any) :: boolean

  defmacro __using__(_) do
    quote do
      require Annex.Data
      alias Annex.Data
      alias Annex.Data.Shape

      @behaviour Annex.Data
    end
  end

  @doc """
  Annex.Data.cast/4 calls cast/3 for an Annex.Data behaviour implementing module.

  Valid shapes are a non-empty tuple of positive integers or the atom `:defer`
  """

  def cast(:defer, data, _) do
    data
  end

  def cast(type, data, {}) when is_list(data) do
    raise ArgumentError,
      message: """
      Annex.Data.cast got an empty tuple for shape.

      Suggestion: use `:defer` to defer casting to another data type or layer.

      type: #{inspect(type)}
      data: #{inspect(data)}
      """
  end

  def cast(:defer, data, _) do
    data
  end

  def cast(type, data, shape) when is_tuple(shape) and is_atom(type) do
    type.cast(data, shape)
  end

  def cast(type, data, :defer) when is_atom(type) do
    type.cast(data, :defer)
  end

  @doc """
  Flattens an Annex.Data into a list of floats.
  """
  @spec to_flat_list(type(), data()) :: Data.flat_data()
  def to_flat_list(:defer, data) when is_list(data), do: List.flatten(data)
  def to_flat_list(:defer, data), do: to_flat_list(:defer, Enum.into(data, []))
  def to_flat_list(type, data), do: type.to_flat_list(data)

  @doc """
  Given an Annex.Data `type` and the `data` returns the shape of the data.

  The shape of data is used to cast between the expected shapes from one Annex.Layer
  to the next or from one Annex.Sequence to the next.
  """
  def shape(:defer, _data), do: :defer
  def shape(type, data), do: type.shape(data)

  @doc """
  Given a type (Data implementing module) and some `data` returns true or false if the
  data is of the correct type.

  Calls `c:is_type?/1` of the `type`.
  """
  def is_type?(:defer, _), do: true
  def is_type?(type, data), do: type.is_type?(data)

  @doc """
  Given a `type`, `data`, and a `target_shape` converts the data to the `type` and `target_shape`

  If the `data` matches the `type` and the `data_shape` matches the `target_shape` the
  data is returned unaltered.

  If either the `type` or `target_shape` do not match the `data` the data is casted using
  `Data.cast/3`.
  """
  def convert(:defer, data, _) do
    data
  end

  def convert(type, data, target_shape) do
    with(
      true <- is_type?(type, data),
      data_shape <- shape(type, data),
      true <- Shape.match?(data_shape, target_shape)
    ) do
      data
    else
      _ -> cast(type, data, target_shape)
    end
  end

  def flat_data_to_tensor(flat_data, shape) when is_tuple(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.reverse()
    |> Enum.reduce(flat_data, fn chunk_size, acc ->
      Enum.chunk_every(acc, chunk_size)
    end)
    |> unwrap()
  end

  defp unwrap([unwrapped]), do: unwrapped
end
