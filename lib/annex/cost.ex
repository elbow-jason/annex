defmodule Annex.Cost do
  alias Annex.Utils

  @type calculate_function :: (any() -> float())
  @type derivative_function :: (any(), any() -> float())
  @type mapping :: %{calculate: calculate_function(), derivative: derivative_function()}

  @callback calculate(any(), any()) :: float()
  @callback derivative(any(), any()) :: float()

  @spec calculate_error(any(), any()) :: list(float())
  def calculate_error(labels, preds), do: Utils.zipmap(labels, preds, fn ax, bx -> ax - bx end)

  @spec calculate(module() | mapping, any(), any()) :: float()
  def calculate(cost, labels, preds), do: get_calculate(cost).(labels, preds)

  @spec derivative(module() | mapping, any(), any()) :: float()
  def derivative(cost, error, data), do: get_derivative(cost).(error, data)

  defp get_calculate(module) when is_atom(module), do: &module.calculate/2
  defp get_calculate(%{calculate: func}) when is_function(func, 2), do: func

  defp get_derivative(module) when is_atom(module), do: &module.derivative/2
  defp get_derivative(%{derivative: func}) when is_function(func, 2), do: func

  #   alias Annex.Utils

  #   @spec by_name(:mse | :rmse) :: (float() -> float())
  #   def by_name(key) do
  #     case key do
  #       :mse -> &mse/1
  #       :rmse -> &rmse/1
  #     end
  #   end

  @spec mse(list(float())) :: float()
  def mse(losses) do
    Utils.mean(losses, fn loss -> :math.pow(loss, 2) end)
  end

  #   @spec rmse(list(float())) :: float()
  #   def rmse(losses) do
  #     losses
  #     |> mse()
  #     |> :math.sqrt()
  #   end

  #   @spec softmax(list(float())) :: list(float())
  #   def softmax(losses) do
  #     maximum = Enum.max(losses)
  #     diffs = Enum.map(losses, fn l -> l - maximum end)
  #     sum_diffs = Enum.sum(diffs)
  #     Enum.map(diffs, fn d -> d / sum_diffs end)
  #   end
end

# end
