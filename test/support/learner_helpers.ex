defmodule Annex.LearnerHelper do
  @moduledoc """
  This module contains utility functions for the testing of an Annex.Learner
  """
  require Logger

  def test_logger(learner, loss, epoch, opts) do
    log_interval = Keyword.get(opts, :log_interval, 10_000)
    inputs = Keyword.get(opts, :inputs)
    labels = Keyword.get(opts, :labels)

    if rem(epoch, log_interval) == 0 do
      Logger.debug(fn ->
        """
        Learner -
        training: #{Keyword.get(opts, :name)}
        epoch: #{epoch}
        loss: #{inspect(loss)}
        inputs: #{inspect(inputs)}
        labels: #{inspect(labels)}
        learner: #{inspect(learner)}
        """
      end)
    end
  end
end
