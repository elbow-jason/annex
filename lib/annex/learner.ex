defmodule Annex.Learner do
  @moduledoc """
  The Learner module defines the types, callbacks, and helper functions for a Learner.

  A Learner is a model that is capable of supervised learning.
  """

  alias Annex.{
    Data,
    Dataset,
    LayerConfig,
    Optimizer,
    Optimizer.SGD
  }

  require Logger

  @type t() :: struct()
  @type options :: Keyword.t()

  @type training_output :: %{atom() => any()}

  @type data :: Data.data()

  @callback init_learner(t(), options()) :: t()

  @callback train(t(), Dataset.t(), options()) :: {t(), training_output()}
  @callback predict(t(), data()) :: data()

  @optional_callbacks [
    train: 3,
    init_learner: 2
  ]

  defmacro __using__(_) do
    quote do
      def __annex__(:learner?), do: true

      @before_compile Annex.Learner
    end
  end

  defmacro __before_compile__(_env) do
    quote do
      def __annex__(_), do: nil
    end
  end

  @spec is_learner?(any) :: boolean()
  def is_learner?(%module{}) do
    is_learner?(module)
  end

  def is_learner?(module) do
    is_atom(module) && function_exported?(module, :__annex__, 1) && module.__annex__(:learner?)
  end

  @spec predict(t(), data()) :: data()
  def predict(%module{} = learner, data) do
    module.predict(learner, data)
  end

  @spec train(t(), Dataset.t(), Keyword.t()) :: {t(), training_output()}
  def train(learner, dataset, opts \\ [])

  def train(%LayerConfig{} = cfg, dataset, opts) do
    cfg
    |> LayerConfig.init_layer()
    |> train(dataset, opts)
  end

  def train(%module{} = learner, dataset, opts) do
    learner
    |> module.init_learner(opts)
    |> do_train(dataset, opts)
  end

  defp debug_logger(_learner, training_output, epoch, opts) do
    log_interval = Keyword.get(opts, :log_interval, 10_000)

    if rem(epoch, log_interval) == 0 do
      Logger.debug(fn ->
        """
        Learner -
        training: #{Keyword.get(opts, :name)}
        epoch: #{epoch}
        training_output: #{inspect(training_output, pretty: true)}
        """
      end)
    end
  end

  defp do_train(%learner_module{} = orig_learner, dataset, opts) do
    {halt_opt, opts} = Keyword.pop(opts, :halt_condition, {:epochs, 1_000})
    {log, opts} = Keyword.pop(opts, :log, &debug_logger/4)

    {optimizer, opts} = Keyword.pop(opts, :optimizer, SGD)
    halt_condition = parse_halt_condition(halt_opt)

    1
    |> Stream.iterate(fn epoch -> epoch + 1 end)
    |> Enum.reduce_while(orig_learner, fn epoch, learner ->
      {%_{} = learner2, training_output} =
        if has_train?(learner_module) do
          learner_module.train(learner, dataset, opts)
        else
          Optimizer.train(optimizer, learner, dataset, opts)
        end

      _ = log.(learner2, training_output, epoch, opts)

      if halt_condition.(learner2, training_output, epoch, opts) do
        {:halt, {learner2, training_output}}
      else
        {:cont, learner2}
      end
    end)
  end

  @spec init_learner(t(), options()) :: t()
  def init_learner(%module{} = learner, options) do
    module.init_learner(learner, options)
  end

  def has_train?(%module{}), do: has_train?(module)
  def has_train?(module) when is_atom(module), do: function_exported?(module, :train, 3)

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
