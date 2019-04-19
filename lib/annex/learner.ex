defmodule Annex.Learner do
  alias Annex.{Utils, Data}
  require Logger

  @type learner :: struct()
  @type options :: Keyword.t()

  @callback init_learner(learner(), options()) :: {:ok, learner()} | {:error, any()}
  @callback train(any(), Data.t(), Data.t(), options()) :: {any(), learner()}
  @callback train_opts(options()) :: options()
  @callback predict(learner(), Data.t()) :: Data.t()

  def train(%module{} = learner, all_inputs, all_labels, opts \\ []) do
    case module.init_learner(learner, opts) do
      {:ok, learner} -> do_train(learner, all_inputs, all_labels, opts)
      {:error, _} = err -> err
    end
  end

  defp debug_logger(_learner, output, epoch, opts) do
    if rem(epoch, 10_000) == 0 do
      Logger.debug(fn ->
        """
        Learner Training #{Keyword.get(opts, :name)}...
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
    |> Enum.reduce_while(learner, fn {{inputs, labels}, epoch}, learner_acc ->
      do_train_reducer(learner_acc, inputs, labels, epoch, halt_condition, log, train_opts)
    end)
  end

  defp do_train_reducer(%module{} = learner1, inputs, label, epoch, halt_condition, log, opts) do
    {output, learner2} = module.train(learner1, inputs, label, opts)
    _ = log.(learner2, output, epoch, opts)

    halt_or_cont =
      if halt_condition.(learner2, output, epoch) do
        :halt
      else
        :cont
      end

    {halt_or_cont, learner2}
  end

  @spec init(learner, options()) :: {:ok, learner()} | {:error, any()}
  def init(%module{} = learner, options) do
    module.init_learner(learner, options)
  end

  defp parse_halt_condition(func) when is_function(func, 3) do
    func
  end

  defp parse_halt_condition({:epochs, num}) when is_number(num) do
    fn _, _, index -> index >= num end
  end

  defp parse_halt_condition({:loss_less_than, num}) when is_number(num) do
    fn _, output, _ -> get_loss(output) < num end
  end

  defp get_loss(total_loss) when is_float(total_loss), do: total_loss
  defp get_loss(%module{} = output), do: module.get_loss(output)
end
