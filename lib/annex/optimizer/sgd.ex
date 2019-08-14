defmodule Annex.Optimizer.SGD do
  @moduledoc """
  The optimizer for mini-batch (or non-batching) stochastic gradient descent.
  """
  alias Annex.{
    AnnexError,
    Cost,
    Data,
    Dataset,
    Defaults,
    Layer,
    Layer.Backprop,
    Learner,
    Optimizer
  }

  import Annex.Utils, only: [is_pos_integer: 1]

  @behaviour Optimizer

  @impl Optimizer

  @spec train(Learner.t(), Dataset.t(), Keyword.t()) :: {Learner.t(), Learner.training_output()}
  def train(%{} = learner, dataset, opts) do
    if Layer.is_layer?(learner) do
      train_layer(learner, dataset, opts)
    else
      raise %AnnexError{
        message: "#{__MODULE__}.train/3 requires that the learner implements the Layer behaviour",
        details: [
          learner: learner,
          options: opts,
          dataset: dataset
        ]
      }
    end
  end

  defp train_layer(orig_layer, dataset, opts) do
    cost = Keyword.get_lazy(opts, :cost, fn -> Defaults.get_defaults(:cost) end)

    batch_size = Keyword.get(opts, :batch_size)
    batched_dataset = batch_dataset(dataset, batch_size)

    Enum.reduce(batched_dataset, {orig_layer, %{}}, fn {inputs, labels}, {layer1, _output} ->
      {%{} = layer2, prediction} = Layer.feedforward(layer1, inputs)

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

  @spec batch_dataset(Dataset.t(), nil | pos_integer) :: Dataset.t()
  def batch_dataset(dataset, nil) do
    Enum.shuffle(dataset)
  end

  def batch_dataset(dataset, batch_size) when is_pos_integer(batch_size) do
    dataset
    |> Enum.shuffle()
    |> Stream.cycle()
    |> Enum.take(batch_size)
    |> Enum.shuffle()
  end
end
