defmodule Annex.Cost.MeanSquaredError do
  alias Annex.{Cost, Utils}

  @behaviour Cost

  def calculate(labels, predictions) do
    labels
    |> Utils.subtract(predictions)
    |> calculate()
  end

  def calculate(error) do
    Utils.mean(error, fn loss -> loss * loss end)
  end

  def derivative(errors, _data, _labels \\ []) do
    -2.0 * Enum.sum(errors)
  end
end
