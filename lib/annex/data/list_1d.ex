defmodule Annex.Data.List1D do
  @moduledoc """
  The Annex.Data.List is the most basic Annex.Data.
  """
  use Annex.Data

  @type t() :: [float(), ...]

  defguard is_list1D(data) when Data.is_flat_data(data)

  @impl Data
  @spec cast(any, {pos_integer()}) :: t()
  def cast(data, {n} = shape) when is_list1D(data) and is_integer(n) do
    elements_count = length(data)

    if elements_count != n do
      raise ArgumentError,
        message: """
        The number of items in the provided data did not match the required number of items of the given
        shape.

        shape: #{inspect(shape)}
        expected_count: #{inspect(n)}
        actual_count: #{inspect(elements_count)}
        data: #{inspect(data)}
        """
    end

    data
  end

  @impl Data
  @spec to_flat_list(t()) :: Data.flat_data()
  def to_flat_list(data) when is_list1D(data), do: data

  @impl Data
  @spec shape(t()) :: Data.shape()
  def shape(data) when is_list1D(data), do: {length(data)}

  @impl Data
  @spec is_type?(Data.data()) :: boolean
  def is_type?(data), do: is_list1D(data)
end
