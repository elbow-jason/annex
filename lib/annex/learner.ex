defmodule Annex.Learner do
  alias Annex.{Utils, Data}
  require Logger

  @type learner :: struct()
  @type options :: Keyword.t()

  @callback init_learner(learner(), options()) :: {:ok, learner()} | {:error, any()}
  @callback train(any(), Data.t(), Data.t(), options()) :: {any(), learner()}
  @callback train_opts(options()) :: options()
  @callback predict(learner(), Data.t()) :: Data.t()

  def predict(%module{} = learner, data) do
    module.predict(learner, data)
  end

  @spec train(struct(), Data.dataset(), Data.dataset(), Keyword.t()) ::
          {:ok, list(float()), struct()} | {:error, any()}
  def train(%module{} = learner, all_inputs, all_labels, opts \\ []) do
    with(
      {:ok, learner} <- module.init_learner(learner, opts),
      {loss, learner2} <- do_train(learner, all_inputs, all_labels, opts)
    ) do
      {:ok, loss, learner2}
    else
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
    |> Enum.reduce_while({nil, learner}, fn {{inputs, labels}, epoch}, {_, learner_acc} ->
      {output, learner2} = module.train(learner_acc, inputs, labels, train_opts)

      _ = log.(learner2, output, epoch, opts)

      halt_or_cont =
        if halt_condition.(learner2, output, epoch, opts) do
          :halt
        else
          :cont
        end

      {halt_or_cont, {output, learner2}}
    end)
  end

  @spec init(learner, options()) :: {:ok, learner()} | {:error, any()}
  def init(%module{} = learner, options) do
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
