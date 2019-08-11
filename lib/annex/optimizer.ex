defmodule Annex.Optimizer do
  @moduledoc """
  The Optimizer Behaviour and context for calling optimizer implementations.

  """

  alias Annex.{Dataset, Learner}

  @type t :: module()

  @callback train(Learner.t(), Dataset.t(), Keyword.t()) ::
              {Learner.t(), Learner.training_output()}

  @spec train(t(), Learner.t(), Dataset.t(), Keyword.t()) ::
          {Learner.t(), Learner.training_output()}
  def train(optimizer, learner, dataset, opts) do
    optimizer.train(learner, dataset, opts)
  end
end
