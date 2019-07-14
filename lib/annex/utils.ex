defmodule Annex.Utils do
  @moduledoc """
  This module holds functions that are not necessarily specific to one
  of Annex's other modules.
  """

  defguard is_pos_integer(x) when is_integer(x) and x > 0

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
  @spec random_weights(integer()) :: list(float())
  def random_weights(n) when n > 0 and is_integer(n) do
    Enum.map(1..n, fn _ -> random_float() end)
  end

  @doc """
  Random unifmormly splits a given `dataset` into two datasets at a given `frequency`.
  """
  def split_dataset(dataset, frequency) when frequency >= 0.0 and frequency <= 1.0 do
    grouped =
      dataset
      |> Enum.shuffle()
      |> Enum.group_by(fn _ -> :rand.uniform() > frequency end)

    {Map.get(grouped, true, []), Map.get(grouped, false, [])}
  end

  @doc """
  A strict zip function in which the two given enumerables *must* be the same size.
  """
  @spec zip([any()], [any()]) :: [{any(), any()}]
  def zip([], []) do
    []
  end

  def zip([a | a_rest], [b | b_rest]) do
    [{a, b} | zip(a_rest, b_rest)]
  end

  def zip(%Stream{} = a, b) do
    a
    |> Enum.into([])
    |> zip(b)
  end

  def zip(a, %Stream{} = b) do
    zip(a, Enum.into(b, []))
  end

  @doc """
  A strict zip-while-mapping function in which the two given enumerables *must* be the same size.
  """
  @spec zipmap([any()], [any()], any()) :: [any()]
  def zipmap([], [], _) do
    []
  end

  def zipmap([a | a_rest], [b | b_rest], func) when is_function(func, 2) do
    [func.(a, b) | zipmap(a_rest, b_rest, func)]
  end

  @spec transpose(any()) :: [[any()]]
  def transpose([]), do: []
  def transpose([[] | _]), do: []

  def transpose(a) do
    [Enum.map(a, &hd/1) | transpose(Enum.map(a, &tl/1))]
  end

  @doc """
  Calculates the average.
  """
  @spec mean(any()) :: float()
  def mean([]), do: 0.0

  def mean(items) do
    {counted, totaled} =
      Enum.reduce(items, {0, 0.0}, fn item, {count, total} ->
        {count + 1, total + item}
      end)

    totaled / counted
  end

  def mean([], _), do: 0.0

  def mean(items, mapper) when is_function(mapper, 1) do
    {counted, totaled} =
      Enum.reduce(items, {0, 0.0}, fn item, {count, total} ->
        {count + 1, mapper.(item) + total}
      end)

    totaled / counted
  end

  @doc """
  Calculates the dot product which is the sum of element-wise multiplication of two enumerables.
  """
  @spec dot(list(float()), list(float())) :: float()
  def dot(a, b) when is_list(a) and is_list(b) do
    a
    |> zip(b)
    |> Enum.reduce(0.0, fn {ax, bx}, total -> ax * bx + total end)
  end

  @doc """
  Turns a list of floats into floats between 0.0 and 1.0 at their respective ratio.
  """
  @spec normalize(list(float())) :: list(float())
  def normalize(data) when is_list(data) do
    {minimum, maximum} = Enum.min_max(data)

    case maximum - minimum do
      0.0 -> Enum.map(data, fn _ -> 1.0 end)
      diff -> Enum.map(data, fn item -> (item - minimum) / diff end)
    end
  end

  @doc """
  Turns a list of floats into their proportions.

  The sum of the output should be approximately 1.0.
  """
  @spec proportions(list(float())) :: list(float())
  def proportions(data) when is_list(data) do
    case Enum.sum(data) do
      0.0 -> Enum.map(data, fn item -> item end)
      sum -> Enum.map(data, fn item -> item / sum end)
    end
  end

  def subtract(a, b) do
    zipmap(a, b, fn ax, bx -> ax - bx end)
  end
end
