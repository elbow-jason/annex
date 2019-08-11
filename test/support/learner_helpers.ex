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
        training_output: #{inspect(training_output)}
        learner: #{inspect(learner)}
        """
      end)
    end
  end
end
