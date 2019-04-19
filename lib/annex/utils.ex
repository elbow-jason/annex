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

  def zipmap([], [], _) do
    []
  end

  def zipmap([a | a_rest], [b | b_rest], func) when is_function(func, 2) do
    [func.(a, b) | zipmap(a_rest, b_rest, func)]
  end

  def transpose([]), do: []
  def transpose([[] | _]), do: []

  def transpose(a) do
    [Enum.map(a, &hd/1) | transpose(Enum.map(a, &tl/1))]
  end

  def mean(items) do
    items
    |> Enum.reduce({0, 0.0}, fn item, {count_acc, total_acc} ->
      {count_acc + 1, total_acc + item}
    end)
    |> case do
      {0, _} -> 0.0
      {count, total} -> total / count
    end
  end

  def dot(a, b) when is_list(a) and is_list(b) do
    a
    |> zip(b)
    |> Enum.reduce(0.0, fn {ax, bx}, total -> ax * bx + total end)
  end

  def normalize(data) when is_list(data) do
    {minimum, maximum} = Enum.min_max(data)
    diff = maximum - minimum
    Enum.map(data, fn item -> (item - minimum) / diff end)
  end

  def proportions(data) when is_list(data) do
    sum = Enum.sum(data)
    Enum.map(data, fn item -> item / sum end)
  end
end
