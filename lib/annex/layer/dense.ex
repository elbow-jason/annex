defmodule Annex.Layer.Dense do
  alias Annex.{Layer, Layer.Backprop, Layer.Dense, Layer.Neuron, Utils}

  @behaviour Layer

  @type t :: %__MODULE__{
          neurons: Data.float_data(),
          rows: non_neg_integer(),
          cols: non_neg_integer()
        }

  defstruct neurons: nil,
            rows: nil,
            cols: nil

  defp get_neurons(%Dense{neurons: neurons}), do: neurons

  defp put_neurons(%Dense{} = dense, neurons) do
    %Dense{dense | neurons: neurons}
  end

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

  @spec feedforward(t(), list(float())) :: {list(float()), t()}
  def feedforward(%Dense{} = layer, inputs) do
    {output, neurons} =
      layer
      |> get_neurons()
      |> Enum.map(fn neuron ->
        neuron = Neuron.feedforward(neuron, inputs)
        {Neuron.get_sum(neuron), neuron}
      end)
      |> Enum.unzip()

    {output, %Dense{layer | neurons: neurons}}
  end

  @spec encoder() :: Annex.Data
  def encoder, do: Annex.Data

  @spec backprop(t(), Backprop.t()) :: {t(), Backprop.t()}
  def backprop(%Dense{} = layer, backprops) do
    learning_rate = Backprop.get_learning_rate(backprops)
    derivative = Backprop.get_derivative(backprops)
    loss_pds = Backprop.get_loss_pds(backprops)
    total_loss_pd = Backprop.get_net_loss(backprops)
    cost_func = Backprop.get_cost_func(backprops)

    {neuron_errors, neurons} =
      layer
      |> get_neurons()
      |> Utils.zip(loss_pds)
      |> Enum.map(fn {neuron, loss_pd} ->
        Neuron.backprop(neuron, total_loss_pd, loss_pd, learning_rate, derivative)
      end)
      |> Enum.unzip()

    next_loss_pds =
      neuron_errors
      |> Utils.transpose()
      |> Enum.map(cost_func)

    {put_neurons(layer, neurons), Backprop.put_loss_pds(backprops, next_loss_pds)}
  end
end
