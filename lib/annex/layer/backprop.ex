defmodule Annex.Layer.Backprop do
  @moduledoc """
  Backprop is a module that contains helper functions for manipulating the
  Keyword list that is passed up through a Sequence during the backprop phase.

  The Backprop module is not a Layer implementing module.
  """

  alias Annex.{
    Defaults
  }

  import Keyword, only: [fetch!: 2]

  @type derivative :: (float() -> float())
  @type cost_func :: (float() -> float())

  @type t :: Keyword.t()

  @spec new(keyword) :: keyword
  def new(opts \\ []), do: Keyword.merge(defaults(), opts)

  def defaults, do: Defaults.get_defaults()

  @spec get_cost_func(t()) :: cost_func()
  def get_cost_func(props), do: fetch!(props, :cost_func)

  @spec get_learning_rate(t()) :: float()
  def get_learning_rate(props), do: fetch!(props, :learning_rate)

  @spec get_derivative(t()) :: derivative()
  def get_derivative(props), do: fetch!(props, :derivative)

  @spec get_negative_gradient(t()) :: float()
  def get_negative_gradient(props), do: fetch!(props, :negative_gradient)

  @spec put_derivative(t(), derivative()) :: t()
  def put_derivative(props, derivative) when is_function(derivative, 1) do
    Keyword.put(props, :derivative, derivative)
  end
end
