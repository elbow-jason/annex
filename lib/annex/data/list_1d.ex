defmodule Annex.Data.List1D do
  @moduledoc """
  The Annex.Data.List is the most basic Annex.Data.
  """
  use Annex.Data

  alias Annex.{
    AnnexError,
    Data.List2D,
    Shape,
    Utils
  }

  import Utils, only: [is_pos_integer: 1]

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

  @doc """
  Generates a list of `n` floats between -1.0 and 1.0.
  """
  @spec new_random(pos_integer()) :: t()
  def new_random(n) when is_pos_integer(n) do
    fn -> Utils.random_float() end
    |> Stream.repeatedly()
    |> Enum.take(n)
  end

  @spec ones(pos_integer()) :: t()
  def ones(n) when is_pos_integer(n) do
    fn -> 1.0 end
    |> Stream.repeatedly()
    |> Enum.take(n)
  end

  @doc """
  Calculates the average of a 1D list.
  """
  @spec mean(any()) :: float()
  def mean([]), do: 0.0

  def mean(items) do
    {counted, totaled} =
      Enum.reduce(items, {0, 0.0}, fn item, {count, total} ->
        {count + 1, total + item}
      end)

    totaled / counted
  end

  def mean([], _), do: 0.0

  @doc """
  Calculates the dot product which is the sum of element-wise multiplication of two enumerables.
  """
  @spec dot(t(), t()) :: float()
  def dot(a, b) when is_list1D(a) and is_list1D(b) do
    a
    |> Utils.zipmap(b, fn ax, bx -> ax * bx end)
    |> Enum.sum()
  end

  @spec transpose(t()) :: List2D.t()
  def transpose(data) when is_list1D(data) do
    Enum.map(data, fn f -> [f] end)
  end

  @doc """
  Turns a list of floats into floats between 0.0 and 1.0 at their respective ratio.
  """
  @spec normalize(t()) :: t()
  def normalize(data) when is_list1D(data) do
    {minimum, maximum} = Enum.min_max(data)

    case maximum - minimum do
      0.0 -> Enum.map(data, fn _ -> 1.0 end)
      diff -> Enum.map(data, fn item -> (item - minimum) / diff end)
    end
  end

  @doc """
  Turns a list of floats into their proportions.

  The sum of the output should be approximately 1.0.
  """
  @spec proportions(t()) :: t()
  def proportions(data) when is_list1D(data) do
    case Enum.sum(data) do
      0.0 -> Enum.map(data, fn item -> item end)
      sum -> Enum.map(data, fn item -> item / sum end)
    end
  end
end
