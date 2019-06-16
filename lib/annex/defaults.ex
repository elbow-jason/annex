defmodule Annex.Defaults do
  @moduledoc """
  Defaults handles the access to compile-time configuration values for
  default values of Annex.
  """

  alias Annex.Cost.MeanSquaredError

  @spec get_defaults :: Keyword.t()
  def get_defaults, do: Application.get_env(:annex, :defaults, [])

  @spec get_defaults(atom, any) :: any
  def get_defaults(key, default \\ nil) do
    Keyword.get(get_defaults(), key, default)
  end

  @spec cost() :: module()
  def cost, do: get_defaults(:cost, MeanSquaredError)

  @spec learning_rate() :: float()
  def learning_rate, do: get_defaults(:learning_rate, 0.05)
end
