defmodule Annex.Layer.Backprop do
  alias Annex.{
    Defaults
  }

  import Keyword, only: [fetch!: 2]

  @type derivative :: (float() -> float())
  @type cost_func :: (float() -> float())

  @type t :: Keyword.t()

  def new(opts \\ []) do
    Keyword.merge(default(), opts)
  end

  def default do
    [
      derivative: Defaults.derivative(),
      cost_func: Defaults.cost(),
      learning_rate: Defaults.learning_rate()
    ]
  end

  @spec get_cost_func(t()) :: cost_func()
  def get_cost_func(props), do: fetch!(props, :cost_func)

  @spec get_learning_rate(t()) :: float()
  def get_learning_rate(props), do: fetch!(props, :learning_rate)

  @spec get_derivative(t()) :: derivative()
  def get_derivative(props), do: fetch!(props, :derivative)

  @spec get_net_loss(t()) :: float()
  def get_net_loss(props), do: fetch!(props, :net_loss)

  @spec put_derivative(t(), derivative()) :: t()
  def put_derivative(props, derivative) when is_function(derivative, 1) do
    Keyword.put(props, :derivative, derivative)
  end
end
