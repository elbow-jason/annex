defmodule Annex.Data.Shape do
  @moduledoc """
  The Shape module encapsulates helper functions for use in determining how to
  cast Data types on behalf of Annex.Layers.
  """
  import Annex.Utils, only: [is_pos_integer: 1]

  @type shape_value :: integer() | :any

  @type shape1 :: {shape_value()}
  @type shape2 :: {shape_value(), shape_value()}
  @type shape3 :: {shape_value(), shape_value(), shape_value()}
  @type t :: shape1 | shape2 | shape3

  defguard is_shape(x) when is_tuple(x) and tuple_size(x) > 0

  @spec match?(t() | nil, t() | nil) :: boolean
  def match?(concrete_shape, abstract_shape)
      when tuple_size(concrete_shape) == tuple_size(abstract_shape) do
    abstract_list = Tuple.to_list(abstract_shape)

    concrete_shape
    |> Tuple.to_list()
    |> Enum.zip(abstract_list)
    |> Enum.all?(fn
      {x, :any} when is_pos_integer(x) -> true
      {x, y} when is_pos_integer(x) and x === y -> true
      _ -> false
    end)
  end

  def match?(_data_shape, _shape), do: false

  def convert_abstract_to_concrete(abstract, concrete) do
    concretify_abstract(abstract, concrete)
  end

  defp concretify_abstract(abstract, matching_concrete) do
    concrete_size = product(matching_concrete)
    abstract_size = factor(abstract)
    remainder = rem(concrete_size, abstract_size)

    case describe(abstract) do
      _ when concrete_size < abstract_size ->
        error =
          concretify_error(
            reason: "abstract_size was larger than concrete_size",
            concrete_size: concrete_size,
            abstract_size: abstract_size,
            abstract: abstract,
            matching_concrete: matching_concrete
          )

        {:error, error}

      :concrete when concrete_size != abstract_size ->
        error =
          concretify_error(
            reason:
              "abstract shape size must exactly match concrete shape size when abstract has no :any",
            concrete_size: concrete_size,
            abstract_size: abstract_size,
            abstract: abstract,
            matching_concrete: matching_concrete
          )

        {:error, error}

      _ when remainder != 0 ->
        error =
          concretify_error(
            reason: "abstract shape size cannot match concrete shape size",
            concrete_size: concrete_size,
            abstract_size: abstract_size,
            abstract: abstract,
            matching_concrete: matching_concrete
          )

        {:error, error}

      _ ->
        new_dimension = div(concrete_size, abstract_size)

        concreted =
          abstract
          |> Tuple.to_list()
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
          |> List.to_tuple()

        {:ok, concreted}
    end
  end

  defp concretify_error(kwargs) do
    message =
      "#{inspect(__MODULE__)} encountered an error while turning an abstract shape to a concrete shape."

    Annex.AnnexError.build(message, kwargs)
  end

  # defp convert_concrete_to_concrete(concrete_source, concrete_target) do
  #   source_size = product(concrete_source)
  #   target_size = product(concrete_target)

  #   if source_size == target_size do
  #     concrete_target
  #   else
  #     raise Annex.AnnexError,
  #       message: """
  #       Failed to convert concrete shapes! Shapes could not be matched.

  #       concrete_source: #{inspect(concrete_source)}
  #       concrete_target: #{inspect(concrete_target)}
  #       target_size: #{inspect(target_size)}
  #       source_size: #{inspect(source_size)}
  #       """
  #   end
  # end

  @spec is_factor_of?(integer | t(), integer) :: boolean
  def is_factor_of?(num, factor) when is_integer(num) and is_integer(factor) do
    rem(num, factor) == 0
  end

  def is_factor_of?(shape, factor) do
    shape
    |> product()
    |> is_factor_of?(factor)
  end

  @spec is_shape?(any) :: boolean
  def is_shape?(tup) when is_tuple(tup) do
    tup
    |> Tuple.to_list()
    |> Enum.all?(&is_shape_value?/1)
  end

  def is_shape?(_) do
    false
  end

  @spec is_shape_value?(any) :: boolean
  def is_shape_value?(n) when is_pos_integer(n), do: true
  def is_shape_value?(:any), do: true
  def is_shape_value?(_), do: false

  @spec product(tuple) :: pos_integer()
  def product(shape) when is_tuple(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.reduce(&Kernel.*/2)
  end

  @spec factor(tuple) :: pos_integer()
  def factor(shape) when is_tuple(shape) and tuple_size(shape) > 0 do
    shape
    |> Tuple.to_list()
    |> Enum.reduce(1, fn
      :any, acc -> acc
      n, acc -> n * acc
    end)
  end

  @spec describe(t()) :: :abstract | :concrete
  def describe(shape) when is_tuple(shape) do
    if all_integers?(shape) do
      :concrete
    else
      :abstract
    end
  end

  defp all_integers?(shape) when is_tuple(shape) and tuple_size(shape) > 0 do
    shape
    |> Tuple.to_list()
    |> Enum.all?(&is_integer/1)
  end

  # @spec resolve(t() | nil, t() | nil) :: {pos_integer, pos_integer}
  # def resolve(nil, nil) do
  #   raise Annex.AnnexError,
  #     message: """
  #     Shape.resolve/1 cannot resolve nils. At least one shape must exist to be resolvable.
  #     """
  # end

  # def resolve(nil, right), do: right
  # def resolve(left, nil), do: left
  # def resolve(left, right), do: {resolve_rows(left), resolve_columns(right)}

  def resolve_rows({_}), do: 1
  def resolve_rows({rows, _}) when is_pos_integer(rows), do: rows

  def resolve_columns({n}) when is_pos_integer(n), do: n
  def resolve_columns({_, columns}) when is_pos_integer(columns), do: columns
end
