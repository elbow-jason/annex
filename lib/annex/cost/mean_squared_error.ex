defmodule Annex.Cost.MeanSquaredError do
  @moduledoc """
  MeanSquaredError is the module that encapsulates the calculation of the mean
  squared error equation.

  The mean squared error is a calculation that describes the penalty (`Cost`)
  of the distance between points that highly penalizes large values and
  slightly penalizes small values by applying the square (`x * x`) of the
  difference between the expected values (`labels`) and the predicted values
  (`predictions`) of a sequence/network/model, summing the squares and taking
  the mean/average.

  MeanSquaredError is the default Cost for Annex. MeanSquaredError enables
  a neural network to more quickly "learn" by associating a higher error
  penalty for higher error values. Basically, the more incorrect a prediction
  the higher the penalty by a power of 2.
  """

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
