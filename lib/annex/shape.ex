defmodule Annex.Shape do
  @moduledoc """
  The Shape module encapsulates helper functions for use in determining the shapes
  and validity of shapes between Layers and Layers; Data and Data; and Data and Layers.
  """
  import Annex.Utils, only: [is_pos_integer: 1]
  alias Annex.AnnexError

  @type shape_any :: :any
  @type concrete_dimension :: pos_integer()
  @type abstract_dimension :: concrete_dimension() | shape_any()
  @type concrete :: [concrete_dimension(), ...]
  @type abstract :: [abstract_dimension(), ...]

  @type t :: concrete | abstract

  defguard is_shape(x) when is_list(x) and (is_integer(hd(x)) or hd(x) == :any)

  @spec convert_abstract_to_concrete(abstract(), concrete()) :: concrete | no_return
  def convert_abstract_to_concrete(abstract, concrete) do
    concretify_abstract(abstract, concrete)
  end

  defp concretify_abstract(abstract, target_concrete) do
    concrete_size = product(target_concrete)
    abstract_size = factor(abstract)
    remainder = rem(concrete_size, abstract_size)

    case kind(abstract) do
      _ when concrete_size < abstract_size ->
        error =
          concretify_error(
            reason: "abstract_size was larger than concrete_size",
            concrete_size: concrete_size,
            abstract_size: abstract_size,
            abstract: abstract,
            target_concrete: target_concrete
          )

        {:error, error}

      :concrete when concrete_size != abstract_size ->
        concretify_error(
          reason:
            "abstract shape size must exactly match concrete shape size when abstract has no :any",
          concrete_size: concrete_size,
          abstract_size: abstract_size,
          abstract: abstract,
          target_concrete: target_concrete
        )

      _ when remainder != 0 ->
        concretify_error(
          reason: "abstract shape size cannot match concrete shape size",
          concrete_size: concrete_size,
          abstract_size: abstract_size,
          abstract: abstract,
          target_concrete: target_concrete
        )

      _ ->
        new_dimension = div(concrete_size, abstract_size)

        abstract
        |> Enum.reduce({new_dimension, []}, fn
          value, {substitute, acc} when is_integer(value) ->
            {substitute, [value | acc]}

          :any, {substitute, acc} ->
            {1, [substitute | acc]}
        end)
        |> case do
          {_, acc} -> acc
        end
        |> Enum.reverse()
    end
  end

  defp concretify_error(details) do
    message =
      "#{inspect(__MODULE__)} encountered an error while turning an abstract shape to a concrete shape."

    raise %AnnexError{message: message, details: details}
  end

  @spec is_factor_of?(integer | t(), integer) :: boolean
  def is_factor_of?(num, factor) when is_integer(num) and is_integer(factor) do
    rem(num, factor) == 0
  end

  def is_factor_of?(shape, factor) when is_shape(shape) do
    shape
    |> product()
    |> is_factor_of?(factor)
  end

  @spec is_shape?(any) :: boolean
  def is_shape?(shape) when is_shape(shape) do
    Enum.all?(shape, &is_shape_value?/1)
  end

  def is_shape?(_) do
    false
  end

  @spec is_shape_value?(any) :: boolean
  def is_shape_value?(n) when is_pos_integer(n), do: true
  def is_shape_value?(:any), do: true
  def is_shape_value?(_), do: false

  @doc """
  Returns the product of a concrete `shape`. Given an abstract `shape` product/1
  will raise an ArithmeticError. The closest related function to product/1 is
  `factor/1`. If you need the product of the integers without the `:any` in the
  shape use `factor/2`.

  For more info about concrete vs abstract shapes see `Shape.kind/1`.
  """
  @spec product(concrete()) :: pos_integer()
  def product(shape) when is_shape(shape) do
    Enum.reduce(shape, &Kernel.*/2)
  end

  @spec factor(t()) :: pos_integer()
  def factor(shape) when is_shape(shape) do
    Enum.reduce(shape, 1, fn
      :any, acc -> acc
      n, acc -> n * acc
    end)
  end

  @doc """
  Given a valid `shape` checks the content of the shape to determine whether
  the `shape` is a `:concrete` kind or an `:abstract` kind of shape.

  A `:concrete` shape contains only positive integers and represents a known,
  exact shape. For example, the shape `[3, 4]` represents a two dimensional
  matrix that has 3 rows and 4 columns. `Data` always has a `:concrete` shape;
  the elements of a `Data` can be counted.

  An `:abstract` shape contains both positive integers and/or `:any`. An
  `:abstract` shape represents a partially unknown shape. For example, the
  shape `[3, :any]` represents a two dimensional shape that has 3 rows and
  any positive integer `n` number of columns.

  Some operations on `Data` can express shape requirements in an `:abstract`
  way.

  The `:abstract` shape idea is particularly useful for describing the
  possible valid shapes for casting data for a shaped `Layer`. For example,
  during feedfoward a `Dense` layer requires that `input` has the same number
  of rows as the Dense layer has columns so that it can perform a matrix dot
  operation. For a Dense layer with 2 rows and 3 columns the shape it demands
  for casting would be `[3, :any]`.
  """
  @spec kind(t()) :: :abstract | :concrete
  def kind(shape) when is_shape(shape) do
    if all_integers?(shape) do
      :concrete
    else
      :abstract
    end
  end

  defp all_integers?(shape) when is_shape(shape) do
    Enum.all?(shape, &is_integer/1)
  end

  def resolve_rows([_]), do: 1
  def resolve_rows([rows, _]) when is_pos_integer(rows), do: rows

  def resolve_columns([n]) when is_pos_integer(n), do: n
  def resolve_columns([_, columns]) when is_pos_integer(columns), do: columns
end
