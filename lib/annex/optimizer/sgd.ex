defmodule Annex.Optimizer.SGD do
  @moduledoc """
  The optimizer for stochastic gradient descent.
  """
  alias Annex.{
    Cost,
    Data,
    Defaults,
    Layer,
    Layer.Backprop
  }

  def train(%{} = learner, dataset, opts) do
    cond do
      Layer.is_layer?(learner) ->
        train_layer(learner, dataset, opts)
    end
  end

  defp train_layer(orig_layer, dataset, opts) do
    batch_size = Keyword.get(opts, :batch_size, 1)
    cost = Keyword.get_lazy(opts, :cost, fn -> Defaults.get_defaults(:cost) end)

    batched_dataset = batch_dataset(dataset, batch_size)

    Enum.reduce(batched_dataset, orig_layer, fn {inputs, labels}, layer1 ->
      {%{} = layer2, prediction} = Layer.feedforward(layer1, inputs)

      prediction = Data.to_flat_list(prediction)
      labels = Data.to_flat_list(labels)

      error = Data.error(prediction, labels)

      props = Backprop.new()
      {layer3, _error2, _props} = Layer.backprop(layer2, error, props)

      loss = Cost.calculate(cost, error)

      output = %{
        loss: loss,
        inputs: inputs,
        labels: labels,
        prediction: prediction,
        error: error
      }

      {layer3, output}
    end)
  end

  def batch_dataset(dataset, batch_size) do
    dataset
    |> Enum.shuffle()
    |> Enum.take(batch_size)
  end
end
