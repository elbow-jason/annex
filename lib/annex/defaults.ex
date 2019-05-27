defmodule Annex.Defaults do
  alias Annex.{
    Cost,
    # Cost.MeanSquaredError,
    Layer.Activation
  }

  @spec cost_func() :: (float() -> float())
  def cost_func do
    case Application.get_env(:annex, :cost_func, &Cost.mse/1) do
      func1 when is_function(func1, 1) ->
        func1

      {module, func} when is_atom(module) and is_atom(func) ->
        fn errors ->
          apply(module, func, [errors])
        end
    end
  end

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
