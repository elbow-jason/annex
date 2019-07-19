defmodule Annex.Data.Shape do
  @moduledoc """
  The Shape module encapsulates helper functions for use in determining how to
  cast Data types on behalf of Annex.Layers.
  """

  @type t :: tuple()
  @type defer :: :defer

  def invert(:defer) do
    :defer
  end

  def invert(tup) when is_tuple(tup) do
    tup
    |> Tuple.to_list()
    |> Enum.reverse()
    |> List.to_tuple()
  end

  def match?(_, :defer) do
    true
  end

  def match?(data_shape, layer_shape) do
    is_shape?(data_shape) and is_shape?(layer_shape) and data_shape === layer_shape
  end

  def is_shape?(:defer), do: true

  def is_shape?(tup) when is_tuple(tup) do
    tup
    |> Tuple.to_list()
    |> Enum.all?(&is_integer/1)
  end

  def is_shape?(_) do
    false
  end

  def product(shape) when is_tuple(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.reduce(fn n, acc -> n * acc end)
  end
end
