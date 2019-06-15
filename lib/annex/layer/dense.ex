defmodule Annex.Layer.Dense do
  @moduledoc """
  `rows` are the number outputs and `columns` are the number of inputs.
  """

  use Annex.Debug
  use Annex.Layer.ListLayer

  alias Annex.{
    Layer,
    Layer.Backprop,
    Layer.Dense,
    Layer.ListLayer,
    Layer.Neuron,
    Utils
  }

  import Annex.Utils, only: [is_pos_integer: 1]

  @behaviour Layer

  @type t :: %__MODULE__{
          neurons: list(Neuron.t()),
          rows: pos_integer(),
          columns: pos_integer() | nil,
          input: list(float()),
          output: list(float()),
          initialized?: boolean()
        }

  defstruct neurons: [],
            rows: nil,
            columns: nil,
            input: [],
            output: [],
            initialized?: false

  defp put_neurons(%Dense{} = dense, neurons) do
    %Dense{dense | neurons: neurons}
  end

  defp get_neurons(%Dense{neurons: neurons}), do: neurons

  defp put_output(%Dense{} = dense, output) when is_list(output) do
    %Dense{dense | output: output}
  end

  defp get_output(%Dense{output: o}) when is_list(o), do: o

  defp put_input(%Dense{} = dense, input) do
    %Dense{dense | input: input}
  end

  defp get_input(%Dense{input: input}) when is_list(input), do: input

  @spec init_layer(t(), Keyword.t()) :: {:ok, t()}
  def init_layer(layer, opts \\ [])

  def init_layer(%Dense{initialized?: false} = layer, _opts) do
    {:ok, %Dense{put_random_neurons(layer) | initialized?: true}}
  end

  def init_layer(%Dense{initialized?: true} = layer, _opts) do
    {:ok, layer}
  end

  def build(rows, columns) when is_pos_integer(rows) and is_pos_integer(columns) do
    %Dense{
      rows: rows,
      columns: columns
    }
  end

  def build(rows) when is_integer(rows) and rows > 0 do
    %Dense{rows: rows}
  end

  def build(rows, columns, weights, biases) do
    debug_assert "Dense rows must be a positive integer", do: is_integer(rows)
    debug_assert "Dense rows must be a positive integer", do: rows > 0
    debug_assert "Dense columns must be a positive integer", do: is_integer(columns)
    debug_assert "Dense columns must be a positive integer", do: columns > 0
    debug_assert "Dense weights must be a list of floats", do: is_list(data)
    debug_assert "Dense weights must be a list of floats", do: Enum.all?(data, &is_float/1)
    debug_assert "Dense biases must be a list of floats", do: is_list(biases)
    debug_assert "Dense biases must be a list of floats", do: Enum.all?(data, &is_float/1)

    neurons =
      weights
      |> Enum.chunk_every(columns)
      |> Enum.zip(biases)
      |> Enum.map(fn {weights, bias} -> Neuron.new(weights, bias) end)

    %Dense{
      neurons: neurons,
      rows: rows,
      columns: columns,
      initialized?: true
    }
  end

  @spec feedforward(t(), ListLayer.t()) :: {t(), ListLayer.t()}
  def feedforward(%Dense{} = layer, input) do
    output =
      layer
      |> get_neurons()
      |> Enum.map(fn neuron -> Neuron.feedforward(neuron, input) end)

    updated_layer =
      layer
      |> put_input(input)
      |> put_output(output)

    {updated_layer, output}
  end

  @spec backprop(t(), ListLayer.t(), Backprop.t()) :: {t(), ListLayer.t(), Backprop.t()}
  def backprop(%Dense{} = layer, error, props) do
    learning_rate = Backprop.get_learning_rate(props)
    derivative = Backprop.get_derivative(props)
    negative_gradient = Backprop.get_negative_gradient(props)

    output = get_output(layer)
    input = get_input(layer)

    {neuron_error, neurons} =
      layer
      |> get_neurons()
      |> Utils.zip(error)
      |> Utils.zip(output)
      |> Enum.map(fn {{neuron, local_error}, neuron_output} ->
        sum_deriv = derivative.(neuron_output)
        Neuron.backprop(neuron, input, sum_deriv, negative_gradient, local_error, learning_rate)
      end)
      |> Enum.unzip()

    next_error =
      neuron_error
      |> Utils.transpose()
      |> Enum.map(&Enum.sum/1)

    {put_neurons(layer, neurons), next_error, props}
  end

  defp random_neurons(rows, columns) do
    Enum.map(1..rows, fn _ -> Neuron.new_random(columns) end)
  end

  defp put_random_neurons(%Dense{rows: rows, columns: columns} = layer) do
    put_neurons(layer, random_neurons(rows, columns))
  end
end
