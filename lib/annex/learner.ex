defmodule Annex.Learner do
  @moduledoc """
  The Learner module defines the types, callbacks, and helper functions for a Learner.

  A Learner is a model that is capable of supervised learning.
  """

  alias Annex.{
    Data,
    LayerConfig,
    Utils
  }

  require Logger

  @type t() :: struct()
  @type options :: Keyword.t()
  @type data :: Data.data()
  @type train_output :: %{atom() => any()}

  @callback init_learner(t(), options()) :: {:ok, t()} | {:error, any()}
  @callback train(t(), data(), data(), options()) :: {t(), train_output()}
  @callback train_opts(options()) :: options()
  @callback predict(t(), data()) :: data()

  @spec predict(t(), data()) :: data()
  def predict(%module{} = learner, data) do
    module.predict(learner, data)
  end

  @spec train(t(), data(), data(), Keyword.t()) :: {:ok, t(), list(float())} | {:error, any()}
  def train(learner, all_inputs, all_labels, opts \\ [])

  def train(%LayerConfig{} = cfg, all_inputs, all_labels, opts) do
    case LayerConfig.init_layer(cfg) do
      {:ok, learner} ->
        train(learner, all_inputs, all_labels, opts)

      {:error, _} = error ->
        error
    end
  end

  def train(%module{} = learner, all_inputs, all_labels, opts) do
    with(
      {:ok, learner} <- module.init_learner(learner, opts),
      {learner2, loss} <- do_train(learner, all_inputs, all_labels, opts)
    ) do
      {:ok, learner2, loss}
    else
      {:error, _} = err -> err
    end
  end

  defp debug_logger(_learner, loss, epoch, opts) do
    log_interval = Keyword.get(opts, :log_interval, 10_000)

    if rem(epoch, log_interval) == 0 do
      Logger.debug(fn ->
        """
        Learner -
        training: #{Keyword.get(opts, :name)}
        epoch: #{epoch}
        loss: #{inspect(loss)}
        """
      end)
    end
  end

  defp do_train(%module{} = learner, all_inputs, all_labels, opts) do
    {halt_opt, opts} = Keyword.pop(opts, :halt_condition, {:epochs, 1_000})
    {log, opts} = Keyword.pop(opts, :log, &debug_logger/4)
    halt_condition = parse_halt_condition(halt_opt)

    train_opts = module.train_opts(opts)

    zipped = Utils.zip(all_inputs, all_labels)

    fn ->
      Enum.random(zipped)
    end
    |> Stream.repeatedly()
    |> Stream.with_index(1)
    |> Enum.reduce_while({learner, nil}, fn {{inputs, labels}, epoch},
                                            {learner_acc, _prev_output} ->
      {%_{} = learner2, loss} = module.train(learner_acc, inputs, labels, train_opts)

      _ = log.(learner2, loss, epoch, [{:inputs, inputs}, {:labels, labels} | opts])

      halt_or_cont =
        if halt_condition.(learner2, loss, epoch, opts) do
          :halt
        else
          :cont
        end

      {halt_or_cont, {learner2, loss}}
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
    fn _, loss, _, _ -> loss < num end
  end
end
