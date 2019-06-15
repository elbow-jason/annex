defmodule Annex.Layer.Dense do
  use Annex.Debug

  alias Annex.{
    Layer,
    Layer.Backprop,
    Layer.Dense,
    Layer.ListLayer,
    Layer.Neuron,
    Utils
  }

  @behaviour Layer

  use ListLayer

  @type t :: %__MODULE__{
          neurons: list(Neuron.t()),
          rows: non_neg_integer(),
          cols: non_neg_integer(),
          input: list(float()) | nil,
          output: list(float()) | nil
        }

  defstruct neurons: nil,
            rows: nil,
            cols: nil,
            input: nil,
            output: nil

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

  def get_input(%Dense{input: input}) when is_list(input), do: input

  @spec init_layer(t(), Keyword.t()) :: {:ok, t()}
  def init_layer(%Dense{} = layer, _opts \\ []) do
    {:ok, initialize_neurons(layer)}
  end

  def build_random(rows, cols) do
    %Dense{
      neurons: random_neurons(rows, cols),
      rows: rows,
      cols: cols
    }
  end

  defp random_neurons(rows, cols) do
    Enum.map(1..rows, fn _ -> Neuron.new_random(cols) end)
  end

  def build_from_data(rows, cols, weights, biases) do
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
      |> Enum.chunk_every(cols)
      |> Enum.zip(biases)
      |> Enum.map(fn {weights, bias} -> Neuron.new(weights, bias) end)

    %Dense{
      neurons: neurons,
      rows: rows,
      cols: cols
    }
  end

  defp initialize_neurons(%Dense{rows: rows, cols: cols} = layer) do
    neurons =
      case get_neurons(layer) do
        nil ->
          random_neurons(rows, cols)

        [%Neuron{} | _] = found ->
          found
      end

    put_neurons(layer, neurons)
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
end
