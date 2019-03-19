defmodule Annex do
  alias Annex.{Sequence, Dense, Activation, Cost}
  require Logger

  def sequence(layers, opts \\ []) when is_list(layers) do
    learning_rate = Keyword.get(opts, :learning_rate, 0.05)

    %Sequence{
      layers: layers,
      learning_rate: learning_rate
    }
  end

  def initialize(%module{} = layer) do
    module.initialize(layer)
  end

  def dense(rows, opts \\ []) do
    %Dense{
      rows: rows,
      cols: Keyword.get(opts, :input_dims),
      neurons: Keyword.get(opts, :data)
    }
  end

  def activation(name) do
    Activation.build(name)
  end

  @doc """
  """
  def train(seq, data, all_y_trues, opts \\ [])

  def train(%Sequence{initialized?: false} = seq, data, all_y_trues, opts) do
    seq
    |> Sequence.initialize()
    |> case do
      {:ok, initialized} ->
        train(initialized, data, all_y_trues, opts)
    end
  end

  def train(%Sequence{} = original_sequence, data, all_y_trues, opts) do
    # learn_rate = 0.1
    # number of times to loop through the entire dataset
    epochs = Keyword.get(opts, :epochs, 1000)
    print_at_epoch = Keyword.get(opts, :print_at_epoch, 500)
    name = Keyword.get(opts, :name, nil)

    [data, all_y_trues]
    |> Enum.zip()
    |> Stream.cycle()
    |> Stream.with_index()
    |> Stream.map(fn {{input, target}, index} ->
      {input, target, index}
    end)
    |> Enum.reduce_while(original_sequence, fn {input, target, epoch}, net_acc ->
      net_acc = Sequence.train_once(net_acc, input, target)

      if rem(epoch, print_at_epoch) == 0 do
        y_preds = Enum.map(data, fn d -> Sequence.predict(net_acc, d) end)
        loss = Cost.mse_loss(all_y_trues, y_preds)

        Logger.debug(fn ->
          """
          Sequence: #{name} -
            epoch: #{inspect(epoch)}
            loss: #{inspect(loss)}
          """
        end)
      end

      if epoch >= epochs do
        {:halt, net_acc}
      else
        {:cont, net_acc}
      end
    end)
  end

  def predict(%Sequence{} = seq, data) do
    Sequence.predict(seq, data)
  end
end
