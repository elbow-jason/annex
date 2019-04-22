defmodule Annex.Defaults do
  alias Annex.{Cost, Activation}

  @spec cost() :: (float() -> float())
  def cost, do: get_func(:cost, &Cost.mse/1)

  @spec derivative() :: (float() -> float())
  def derivative, do: get_func(:derivative, &Activation.sigmoid_deriv/1)

  @spec learning_rate() :: float()
  def learning_rate, do: Application.get_env(:annex, :learning_rate, 0.05)

  defp get_func(key, default) do
    case Application.get_env(:annex, key, default) do
      {module, func} ->
        fn val -> apply(module, func, [val]) end

      func when is_function(func, 1) ->
        func
    end
  end
end
