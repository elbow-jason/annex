defmodule Annex.Layer.Dense do
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
          input: list(float()),
          output: list(float())
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
    len_weights = length(weights)
    len_biases = length(biases)

    cond do
      !is_integer(rows) or rows <= 0 ->
        raise ArgumentError,
          message: """
          Rows must be a positive integer -
          rows: #{inspect(rows)}
          """

      !is_integer(cols) or cols <= 0 ->
        raise ArgumentError,
          message: """
          Columns must be a positive integer -
          columns: #{inspect(cols)}
          """

      len_weights != rows * cols ->
        raise ArgumentError,
          message: """
          Bad weights dimensions -
          rows:     #{inspect(rows)}
          cols:     #{inspect(cols)}
          length:   #{inspect(len_weights)}
          expected: #{inspect(rows * cols)}
          weights:  #{inspect(weights)}
          """

      # {:error,
      #  {:bad_data_dimensions, %{rows: rows, cols: cols, len_data: len_data, data: data}}}

      len_biases != rows ->
        raise ArgumentError,
          message: """
          Bad biases length -
          rows:   #{inspect(rows)}
          length: #{inspect(len_biases)}
          biases: #{inspect(biases)}
          """

      Enum.all?(weights, &is_float/1) ->
        raise ArgumentError,
          message: """
          Weights must be floats -
          weights: #{inspect(weights)}
          """

      Enum.all?(biases, &is_float/1) ->
        raise ArgumentError,
          message: """
          Biases must be floats -
          biases: #{inspect(biases)}
          """

      true ->
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
  def backprop(%Dense{} = layer, losses, props) do
    learning_rate = Backprop.get_learning_rate(props)
    derivative = Backprop.get_derivative(props)
    total_loss_pd = Backprop.get_net_loss(props)
    cost_func = Backprop.get_cost_func(props)

    output = get_output(layer)
    input = get_input(layer)

    {neuron_errors, neurons} =
      layer
      |> get_neurons()
      |> Utils.zip(losses)
      |> Utils.zip(output)
      |> Enum.map(fn {{neuron, loss_pd}, neuron_output} ->
        sum_deriv = derivative.(neuron_output)
        Neuron.backprop(neuron, input, sum_deriv, total_loss_pd, loss_pd, learning_rate)
      end)
      |> Enum.unzip()

    next_loss_pds =
      neuron_errors
      |> Utils.transpose()
      |> Enum.map(cost_func)

    {put_neurons(layer, neurons), next_loss_pds, props}
  end
end
