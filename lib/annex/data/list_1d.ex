defmodule Annex.Data.List1D do
  @moduledoc """
  The Annex.Data.List is the most basic Annex.Data.
  """
  use Annex.Data

  alias Annex.{
    AnnexError,
    Shape,
    Utils
  }

  @type t() :: [float(), ...]

  defguard is_list1D(data) when Data.is_flat_data(data)

  @impl Data
  @spec cast(any, Shape.concrete()) :: t()

  def cast(data, [1, n]), do: cast(data, [n])
  def cast(data, [n, 1]), do: cast(data, [n])

  def cast(data, [n] = shape) when is_list1D(data) and is_integer(n) do
    elements_count = length(data)

    if elements_count != n do
      raise %AnnexError{
        message: """
        The number of items in the provided data did not match the required number of items of the given
        shape.
        """,
        details: [
          shape: shape,
          expected_count: n,
          actual_count: elements_count,
          data: data
        ]
      }
    end

    data
  end

  @impl Data
  @spec to_flat_list(t()) :: Data.flat_data()
  def to_flat_list(data) when is_list1D(data), do: data

  @impl Data
  @spec shape(t()) :: Shape.t()
  def shape(data) when is_list1D(data), do: [length(data)]

  @impl Data
  @spec is_type?(Data.data()) :: boolean
  def is_type?(data), do: is_list1D(data)

  @impl Data
  @spec apply_op(t(), Data.op(), Data.args()) :: Data.flat_data()
  def apply_op(data, op, args) do
    case {op, args} do
      {:map, [func]} -> Enum.map(data, func)
      {:subtract, [right]} -> subtract(data, right)
    end
  end

  @spec subtract(t(), t()) :: t()
  def subtract(a, b) do
    Utils.zipmap(a, b, fn ax, bx -> ax - bx end)
  end
end
