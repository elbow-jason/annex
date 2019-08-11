defmodule Annex.Dataset do
  @moduledoc """
  A
  """
  alias Annex.{Data, Utils}

  @type inputs :: Data.data()
  @type labels :: Data.data()
  @type row :: {inputs(), labels()}
  @type t :: [row(), ...]

  def zip(inputs, labels) do
    Utils.zip(inputs, labels)
  end

  def randomize(dataset) do
    Enum.shuffle(dataset)
  end

  @doc """
  Random unifmormly splits a given `dataset` into two datasets at a given `frequency`.

  Very useful for splitting traiings
  """
  def split(dataset, frequency) when frequency >= 0.0 and frequency <= 1.0 do
    %{true => left, false => right} =
      dataset
      |> randomize()
      |> Enum.group_by(fn _ -> :rand.uniform() > frequency end)
      |> Enum.into(%{true => [], false => []})

    {left, right}
  end
end
