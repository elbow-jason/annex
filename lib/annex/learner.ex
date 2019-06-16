defmodule Annex.Learner do
  @moduledoc """
  The Learner module defines the types, callbacks, and helper functions for a Learner.

  A Learner is a model that is capable of supervised learning.
  """

  alias Annex.{Utils}
  require Logger

  @type t() :: t()
  @type options :: Keyword.t()
  @type data :: any()

  @callback init_learner(t(), options()) :: {:ok, t()} | {:error, any()}
  @callback train(t(), data(), data(), options()) :: {t(), data()}
  @callback train_opts(options()) :: options()
  @callback predict(t(), data()) :: data()

  @spec predict(t(), data()) :: data()
  def predict(%module{} = learner, data) do
    module.predict(learner, data)
  end

  @spec train(t(), data(), data(), Keyword.t()) ::
          {:ok, t(), list(float())} | {:error, any()}
  def train(%module{} = learner, all_inputs, all_labels, opts \\ []) do
    with(
      {:ok, learner} <- module.init_learner(learner, opts),
      {learner2, loss} <- do_train(learner, all_inputs, all_labels, opts)
    ) do
      {:ok, learner2, loss}
    else
      {:error, _} = err -> err
    end
  end

  defp debug_logger(_learner, output, epoch, opts) do
    if rem(epoch, 10_000) == 0 do
      Logger.debug(fn ->
        """
        Learner -
        training: #{Keyword.get(opts, :name)}
        epoch: #{epoch}
        loss: #{get_loss(output)}
        """
      end)
    end
  end

  defp do_train(%module{} = learner, all_inputs, all_labels, opts) do
    {halt_opt, opts} = Keyword.pop(opts, :halt_condition, {:epochs, 1_000})
    {log, opts} = Keyword.pop(opts, :log, &debug_logger/4)
    halt_condition = parse_halt_condition(halt_opt)

    train_opts = module.train_opts(opts)

    all_inputs
    |> Utils.zip(all_labels)
    |> Stream.cycle()
    |> Stream.with_index(1)
    |> Enum.reduce_while({learner, nil}, fn {{inputs, labels}, epoch},
                                            {learner_acc, _prev_output} ->
      {%_{} = learner2, output} = module.train(learner_acc, inputs, labels, train_opts)

      _ = log.(learner2, output, epoch, opts)

      halt_or_cont =
        if halt_condition.(learner2, output, epoch, opts) do
          :halt
        else
          :cont
        end

      {halt_or_cont, {learner2, output}}
    end)
  end

  @spec init_learner(t(), options()) :: {:ok, t()} | {:error, any()}
  def init_learner(%module{} = learner, options) do
    module.init_learner(learner, options)
  end

  defp parse_halt_condition(func) when is_function(func, 4) do
    func
  end

  defp parse_halt_condition({:epochs, num}) when is_number(num) do
    fn _, _, index, _ -> index >= num end
  end

  defp parse_halt_condition({:loss_less_than, num}) when is_number(num) do
    fn _, output, _, _ -> get_loss(output) < num end
  end

  defp get_loss(total_loss) when is_float(total_loss), do: total_loss
  defp get_loss(%module{} = output), do: module.get_loss(output)
end
