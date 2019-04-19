defmodule Annex.Cost do
  def by_name(key) do
    case key do
      :mse -> &mse/1
      :rmse -> &rmse/1
    end
  end

  def mse(losses) do
    {count, total} =
      Enum.reduce(losses, {0, 0.0}, fn loss, {count, total} ->
        squared_error = :math.pow(loss, 2)
        {count + 1, squared_error + total}
      end)

    total / count
  end

  def rmse(losses) do
    losses
    |> mse()
    |> :math.sqrt()
  end
end
