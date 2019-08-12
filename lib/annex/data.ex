defmodule Annex.Data do
  @moduledoc """
  Annex.Data defines the callbacks and helpers for data structures used
  by Annex.

  An implementer of the Annex.Layer behaviour must return an Annex.Data
  implementer from the `c:data_type/0` callback.
  """
  alias Annex.{
    AnnexError,
    Data,
    Data.List1D,
    Data.List2D,
    Shape
  }

  require Shape

  @typedoc """
  A module that implements the Annex.Data Behaviour.
  """
  @type type :: module()
  @type flat_data :: [float(), ...]
  @type data :: struct() | flat_data() | [flat_data()]
  @type op :: any()
  @type args :: list(any())

  defguard is_flat_data(data) when is_list(data) and is_float(hd(data))

  @callback cast(data, Shape.t()) :: data()
  @callback to_flat_list(data) :: list(float())
  @callback shape(data) :: Shape.t()
  @callback is_type?(any) :: boolean
  @callback apply_op(data(), op(), args()) :: data()

  defmacro __using__(_) do
    quote do
      require Annex.Data
      require Annex.Shape

      alias Annex.Data
      alias Annex.Shape

      @behaviour Annex.Data
    end
  end

  @doc """
  Annex.Data.cast/4 calls cast/3 for an Annex.Data behaviour implementing module.

  Valid shapes are a non-empty tuple of positive integers or any the atom :any.
  e.g. `{2, 3}` or `{3, :any}`
  """
  def cast(type, data, []) when is_list(data) do
    message = "Annex.Data.cast/3 got an empty list for shape"
    raise AnnexError.build(message, type: type, data: data)
  end

  def cast(type, data, shape) when Shape.is_shape(shape) and is_atom(type) do
    type.cast(data, shape)
  end

  @spec cast(Data.data(), Shape.t()) :: Data.data()
  def cast(data, shape) do
    data
    |> infer_type()
    |> cast(data, shape)
  end

  @doc """
  Flattens an Annex.Data into a list of floats via the type's callback.
  """
  @spec to_flat_list(type(), data()) :: Data.flat_data()
  def to_flat_list(type, data), do: type.to_flat_list(data)

  @doc """
  Flattens an Annex.Data into a list of floats via Enum.into/2.
  """
  @spec to_flat_list(Data.data()) :: Data.flat_data()
  def to_flat_list(data) do
    data
    |> infer_type()
    |> to_flat_list(data)
  end

  @doc """
  Given an Annex.Data `type` and the `data` returns the shape of the data.

  The shape of data is used to cast between the expected shapes from one Annex.Layer
  to the next or from one Annex.Sequence to the next.
  """
  @spec shape(type(), data()) :: Shape.t()
  def shape(type, data), do: type.shape(data)

  @spec shape(data()) :: Shape.t()
  def shape(data), do: data |> infer_type() |> shape(data)

  @doc """
  Given a type (Data implementing module) and some `data` returns true or false if the
  data is of the correct type.

  Calls `c:is_type?/1` of the `type`.
  """
  def is_type?(nil, _), do: false
  def is_type?(type, data), do: type.is_type?(data)

  @doc """
  Given a `type`, `data`, and a `target_shape` converts the data to the `type` and `target_shape`

  If the `data` matches the `type` and the `data_shape` matches the `target_shape` the
  data is returned unaltered.

  If either the `type` or `target_shape` do not match the `data` the data is casted using
  `Data.cast/3`.
  """
  def convert(type, data, target_shape) do
    if is_type?(type, data) do
      data_shape = shape(type, data)
      do_convert(type, data, data_shape, target_shape)
    else
      flat = Data.to_flat_list(data)
      data_shape = List1D.shape(flat)
      do_convert(type, flat, data_shape, target_shape)
    end
  end

  defp do_convert(type, data, data_shape, target_shape) do
    new_shape = Shape.convert_abstract_to_concrete(target_shape, data_shape)
    cast(type, data, new_shape)
  end

  def flat_data_to_tensor(flat_data, shape) when Shape.is_shape(shape) do
    shape
    |> Enum.reverse()
    |> Enum.reduce(flat_data, fn chunk_size, acc ->
      Enum.chunk_every(acc, chunk_size)
    end)
    |> unwrap()
  end

  defp unwrap([unwrapped]), do: unwrapped

  @spec infer_type(data()) :: any
  def infer_type(%module{} = item) do
    if function_exported?(module, :data_type, 1) do
      module.data_type(item)
    else
      module
    end
  end

  def infer_type(data) when is_flat_data(data) do
    List1D
  end

  def infer_type([row | _]) when is_flat_data(row) do
    List2D
  end

  @spec apply_op(data(), any, list(any)) :: data
  def apply_op(data, name, args) do
    data
    |> infer_type()
    |> apply_op(data, name, args)
  end

  @spec apply_op(module, data(), any, list(any)) :: data()
  def apply_op(type, data, name, args) when is_atom(type) do
    type.apply_op(data, name, args)
  end

  @spec error(Data.data(), Data.data()) :: Data.flat_data()
  def error(outputs, labels) do
    labels = Data.to_flat_list(labels)

    outputs
    |> Data.to_flat_list()
    |> Data.apply_op(:subtract, [labels])
  end
end
