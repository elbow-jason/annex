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

  defp initialize_neurons(%Dense{rows: rows, cols: cols} = layer) do
    neurons =
      case get_neurons(layer) do
        nil ->
          Enum.map(1..rows, fn _ -> Neuron.new_random(cols) end)

        found when is_list(found) ->
          found
      end

    put_neurons(layer, neurons)
  end

  @spec feedforward(t(), ListLayer.t()) :: {t(), ListLayer.t()}
  def feedforward(%Dense{} = layer, input) do
    {output, neurons} =
      layer
      |> get_neurons()
      |> Enum.map(fn neuron ->
        neuron = Neuron.feedforward(neuron, input)
        {Neuron.get_sum(neuron), neuron}
      end)
      |> Enum.unzip()

    updated_layer =
      layer
      |> put_input(input)
      |> put_output(output)
      |> put_neurons(neurons)

    {updated_layer, output}
  end

  @spec backprop(t(), ListLayer.t(), Backprop.t()) :: {t(), ListLayer.t(), Backprop.t()}
  def backprop(%Dense{} = layer, losses, props) do
    learning_rate = Backprop.get_learning_rate(props)
    derivative = Backprop.get_derivative(props)
    total_loss_pd = Backprop.get_net_loss(props)
    cost_func = Backprop.get_cost_func(props)

    {neuron_errors, neurons} =
      layer
      |> get_neurons()
      |> Utils.zip(losses)
      |> Enum.map(fn {neuron, loss_pd} ->
        Neuron.backprop(neuron, total_loss_pd, loss_pd, learning_rate, derivative)
      end)
      |> Enum.unzip()

    next_loss_pds =
      neuron_errors
      |> Utils.transpose()
      |> Enum.map(cost_func)

    {put_neurons(layer, neurons), next_loss_pds, props}
  end
end
