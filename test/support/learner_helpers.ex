defmodule Annex.LearnerHelper do
  @moduledoc """
  This module contains utility functions for the testing of an Annex.Learner
  """
  require Logger

  def test_logger(learner, training_output, epoch, opts) do
    log_interval = Keyword.get(opts, :log_interval, 1000)

    if rem(epoch, log_interval) == 0 do
      Logger.debug(fn ->
        """
        Learner -
        learner_name: #{Keyword.get(opts, :name)}
        epoch: #{epoch}
        training_output: #{inspect(training_output, pretty: true)}
        learner: #{inspect(learner)}
        """
      end)
    end
  end
end

defmodule Annex.FakeLearnerWithoutTrain do
  use Annex.Learner
  alias Annex.FakeLearnerWithoutTrain

  defstruct thing: 1

  def predict(%FakeLearnerWithoutTrain{} = learner) do
    prediction = [1.0]
    {learner, prediction}
  end
end

defmodule Annex.FakeLearnerWithTrain do
  use Annex.Learner
  alias Annex.FakeLearnerWithTrain

  defstruct thing: 1

  def predict(%FakeLearnerWithTrain{} = learner) do
    prediction = [1.0]
    {learner, prediction}
  end

  def train(learner, dataset, opts) do
    {learner, dataset, opts}
  end
end
