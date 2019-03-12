defmodule Annex.Cost do
  alias Annex.Utils

  def by_name(key) do
    case key do
      :mse -> &mse_loss/2
      :rmse -> &rmse_loss/2
    end
  end

  def mse_loss(y_true, y_pred) do
    flat_y_pred = List.flatten(y_pred)

    {count, total} =
      y_true
      |> List.flatten()
      |> Utils.zip(flat_y_pred)
      |> Enum.reduce({0, 0.0}, fn {y_true_item, y_pred_item}, {count, total} ->
        squared_error = :math.pow(y_true_item - y_pred_item, 2)
        {count + 1, squared_error + total}
      end)

    total / count
  end

  def rmse_loss(y_true, y_pred) do
    y_true
    |> mse_loss(y_pred)
    |> :math.sqrt()
  end
end
