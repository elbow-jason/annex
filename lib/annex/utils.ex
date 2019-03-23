defmodule Annex.Utils do
  @moduledoc """
  This module holds functions that are not necessarily specific to one
  of Annex's other modules.
  """

  @doc """
  Generates a float between -1.0 and 1.0
  """
  @spec random_float() :: float()
  def random_float do
    :rand.uniform() * 2.0 - 1.0
  end

  @doc """
  Generates a list of `n` floats between -1.0 and 1.0.
  """
  @spec random_weights(integer()) :: [float(), ...]
  def random_weights(n) when n > 0 and is_integer(n) do
    Enum.map(1..n, fn _ -> random_float() end)
  end

  def split_dataset(dataset, ratio) when ratio >= 0.0 and ratio <= 1.0 do
    grouped =
      dataset
      |> Enum.shuffle()
      |> Enum.group_by(fn _ -> :rand.uniform() > ratio end)

    {Map.get(grouped, true, []), Map.get(grouped, false, [])}
  end

  def zip([], []) do
    []
  end

  def zip([a | a_rest], [b | b_rest]) do
    [{a, b} | zip(a_rest, b_rest)]
  end
end
