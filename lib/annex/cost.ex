defmodule Annex.Cost do
  @spec by_name(:mse | :rmse) :: (float() -> float())
  def by_name(key) do
    case key do
      :mse -> &mse/1
      :rmse -> &rmse/1
    end
  end

  @spec mse(list(float())) :: float()
  def mse(losses) do
    {count, total} =
      Enum.reduce(losses, {0, 0.0}, fn loss, {count, total} ->
        squared_error = :math.pow(loss, 2)
        {count + 1, squared_error + total}
      end)

    total / count
  end

  @spec rmse(list(float())) :: float()
  def rmse(losses) do
    losses
    |> mse()
    |> :math.sqrt()
  end

  @spec softmax(list(float())) :: list(float())
  def softmax(losses) do
    maximum = Enum.max(losses)
    diffs = Enum.map(losses, fn l -> l - maximum end)
    sum_diffs = Enum.sum(diffs)
    Enum.map(diffs, fn d -> d / sum_diffs end)
  end
end
