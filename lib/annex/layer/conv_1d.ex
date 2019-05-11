defmodule Annex.Layer.Convolution do
  alias Annex.{
    Layer,
    Layer.Backprop,
    Layer.Convolution,
    Utils
  }

  @type filter :: list(float) | list(filter())

  @type t :: %__MODULE__{
          filter: filter(),
          dimensions: list(pos_integer()),
          initialized?: boolean()
        }

  defstruct filter: [],
            dimensions: [],
            initialized?: false

  @behaviour Layer

  def get_filter(%Convolution{filter: filter}), do: filter
  def get_dimensions(%Convolution{dimensions: dimensions}), do: dimensions

  def encoder, do: Annex.Data

  @spec init_layer(t(), Keyword.t()) :: {:ok, t()}
  def init_layer(%Convolution{initialized?: false, dimensions: dimensions} = layer, _) do
    initialized_layer = %Convolution{
      layer
      | filter: random_weights(dimensions),
        initialized?: true
    }

    {:ok, initialized_layer}
  end

  def init_layer(%Convolution{initialized?: true} = layer, _) do
    {:ok, layer}
  end

  defp random_weights([dimension]) when is_integer(dimension), do: random_weights(dimension)
  defp random_weights([dimension | rest]), do: random_weights(dimension, rest)
  defp random_weights([]), do: []
  defp random_weights(n) when is_integer(n), do: Utils.random_weights(n)
  defp random_weights(n, rest), do: Enum.map(1..n, fn _ -> random_weights(rest) end)

  def feedforward(%Convolution{} = layer, inputs) do
    filter = get_filter(layer)

    outputs =
      layer
      |> get_dimensions()
      |> convolve(filter, inputs, [])

    {outputs, layer}
  end

  def convolve(_, _, [], acc) do
    Enum.reverse(acc)
  end

  def convolve([_] = dimensions, filter, [_ | rest] = inputs, acc) do
    result =
      dimensions
      |> slice(inputs)
      |> Utils.zipmap(filter, fn ix, fx -> ix * fx end)
      |> Enum.sum()

    convolve(dimensions, filter, rest, [result | acc])
  end

  defp slice([d1], items) do
    items
    |> Enum.slice(0, d1)
    |> extend(d1, 0.0)
  end

  defp extend(items, size, value) do
    diff = size - length(items)

    if diff != 0 do
      items ++ Enum.map(1..diff, fn _ -> value end)
    else
      items
    end
  end

  def backprop(%Convolution{} = layer, %Backprop{} = backprop) do
    {layer, backprop}
  end
end
