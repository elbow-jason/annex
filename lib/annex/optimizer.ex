defmodule Annex.Optimizer do
  def batch_data(%optimizer_module{} = optimizer, dataset) do
    if function_exported?(optimizer_module, :batch_data, 1) do
      optimizer_module.batch_data(optimizer, dataset)
    else
      dataset
    end
  end

  def train(optimizer, learner, dataset, opts) do
    optimizer.train(learner, dataset, opts)
  end
end
