defmodule Annex.Cost.MeanSquaredError do
  alias Annex.{Cost, Utils}

  @behaviour Cost

  def calculate(labels, predictions) do
    labels
    |> Utils.subtract(predictions)
    |> Utils.mean(fn loss -> :math.pow(loss, 2) end)
  end

  def derivative(_errors, _data) do
    raise "not implemented"
  end
end
