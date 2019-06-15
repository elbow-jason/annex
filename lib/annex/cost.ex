defmodule Annex.Cost do
  @type calculate_function :: (any() -> float())
  @type derivative_function :: (any(), any(), any() -> float())
  @type mapping :: %{calculate: calculate_function(), derivative: derivative_function()}

  @callback calculate(any()) :: float()
  @callback derivative(any(), any(), any()) :: float()

  @spec calculate(module() | mapping, any()) :: float()
  def calculate(cost, error), do: get_calculate(cost).(error)

  @spec derivative(module() | mapping, any(), any(), any()) :: float()
  def derivative(cost, error, data, labels), do: get_derivative(cost).(error, data, labels)

  defp get_calculate(module) when is_atom(module), do: &module.calculate/1
  defp get_calculate(%{calculate: func}) when is_function(func, 1), do: func

  defp get_derivative(module) when is_atom(module), do: &module.derivative/3
  defp get_derivative(%{derivative: func}) when is_function(func, 3), do: func
end
