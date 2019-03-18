defmodule Annex.Network do
  alias Annex.{Network, Layer, Cost}
  require Logger

  defstruct layers: [],
            learn_rate: 0.05

  def get_layers(%Network{layers: layers}), do: layers
  def get_learn_rate(%Network{learn_rate: lr}), do: lr

  def predict(%Network{} = net, inputs) do
    {y_pred, _} = feedforward(net, inputs)
    y_pred
  end

  def feedforward(%Network{} = net, inputs) do
    {output, layers} =
      net
      |> get_layers()
      |> Enum.reduce({inputs, []}, fn layer, {input, layers} ->
        {out, fed_layer} = Layer.feedforward(layer, input)
        {out, [fed_layer | layers]}
      end)
      |> case do
        {output, rev_layers} ->
          {output, Enum.reverse(rev_layers)}
      end

    {output, %Network{net | layers: layers}}
  end

  def backprop(%Network{} = net, total_loss_pd, loss_pd) do
    learn_rate = get_learn_rate(net)

    {_, layers} =
      net
      |> get_layers()
      |> Enum.reverse()
      |> Enum.reduce({loss_pd, []}, fn layer, {prev_loss_pd, layers} ->
        activation_deriv = Layer.get_activation_deriv(layer)

        {next_loss_pd, layer} =
          Layer.backprop(layer, total_loss_pd, prev_loss_pd, learn_rate, activation_deriv)

        {next_loss_pd, [layer | layers]}
      end)

    %Network{net | layers: layers}
  end

  @doc """
  """
  def train(%Network{} = orig_net, data, all_y_trues, opts \\ []) do
    # learn_rate = 0.1
    # number of times to loop through the entire dataset
    epochs = Keyword.get(opts, :epochs, 1000)
    print_at_epoch = Keyword.get(opts, :print_at_epoch, 500)
    name = Keyword.get(opts, :network_name, nil)

    [data, all_y_trues]
    |> Enum.zip()
    |> Stream.cycle()
    |> Stream.with_index()
    |> Stream.map(fn {{input, target}, index} ->
      {input, target, index}
    end)
    |> Enum.reduce_while(orig_net, fn {input, target, epoch}, net_acc ->
      net_acc = train_once(net_acc, input, target)

      if rem(epoch, print_at_epoch) == 0 do
        y_preds = Enum.map(data, fn d -> Network.predict(net_acc, d) end)
        loss = Cost.mse_loss(all_y_trues, y_preds)

        Logger.debug(fn ->
          """
          Neural Network #{name} -
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

  defp calc_total_error(net_outputs, labels) do
    [labels, net_outputs]
    |> Enum.zip()
    |> Enum.map(fn
      {y_true_item, y_pred_item} -> y_true_item - y_pred_item
    end)
  end

  defp train_once(%Network{} = net1, x, y_true) do
    {y_pred, net2} = Network.feedforward(net1, x)
    network_error = calc_total_error(y_pred, y_true)
    total_loss_pd = -2 * Enum.sum(network_error)
    ones = Enum.map(network_error, fn _ -> 1.0 end)
    Network.backprop(net2, total_loss_pd, ones)
  end
end
