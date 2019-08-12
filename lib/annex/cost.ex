defmodule Annex.Cost do
  @moduledoc """
  The Cost module defines the types, callbacks, and helper functions for calculating
  a the loss and gradient of the lossd, for gradient descent, of a network.

  The default Cost for Annex is Annex.Cost.MeanSquaredError.
  """
  alias Annex.Data

  @type cost_function :: (any() -> float())

  @callback calculate(any()) :: float()

  @type t :: module()

  @spec calculate(t(), Data.data()) :: float()
  def calculate(cost, error), do: get_calculate(cost).(error)

  defp get_calculate(module) when is_atom(module), do: &module.calculate/1
end
