defmodule Annex.Sequence do
  alias Annex.{Sequence, Layer}
  @behaviour Layer

  defstruct layers: [],
            learning_rate: 0.05,
            initialized?: false

  def get_layers(%Sequence{layers: layers}) do
    layers
  end

  def get_learning_rate(%Sequence{learning_rate: learning_rate}) do
    learning_rate
  end

  def initialize(%Sequence{initialized?: true} = seq) do
    {:ok, seq}
  end

  def initialize(%Sequence{initialized?: false} = seq1) do
    seq1
    |> get_layers()
    |> Enum.map(&Layer.initialize/1)
    |> Enum.group_by(
      fn {status, _} -> status end,
      fn {_, initialized} -> initialized end
    )
    |> case do
      %{error: errors} ->
        {:error, errors}

      %{ok: initialized_layers} ->
        initialized_seq = %Sequence{seq1 | layers: initialized_layers, initialized?: true}

        {:ok, initialized_seq}
    end
  end

  def feedforward(%Sequence{} = seq, inputs) do
    {output, layers} =
      seq
      |> get_layers()
      |> Enum.reduce({inputs, []}, fn layer, {input, layers} ->
        {out, fed_layer} = Layer.feedforward(layer, input)
        {out, [fed_layer | layers]}
      end)
      |> case do
        {output, rev_layers} ->
          {output, Enum.reverse(rev_layers)}
      end

    {output, %Sequence{seq | layers: layers}}
  end

  def predict(%Sequence{} = seq, data) do
    {prediction, _} = feedforward(seq, data)
    prediction
  end

  def backprop(%Sequence{} = seq, total_loss_pd, loss_pd, _) do
    learning_rate = get_learning_rate(seq)

    {next_layers_loss_pd, next_layer_opts, layers} =
      seq
      |> get_layers()
      |> Enum.reverse()
      |> Enum.reduce({loss_pd, [], []}, fn layer, {prev_loss_pd, prev_layer_opts, layers} ->
        layer_opts = [{:learning_rate, learning_rate} | prev_layer_opts]

        {next_backprop_data, next_layer_opts, layer} =
          Layer.backprop(layer, total_loss_pd, prev_loss_pd, layer_opts)

        {next_backprop_data, next_layer_opts, [layer | layers]}
      end)

    {next_layers_loss_pd, next_layer_opts, %Sequence{seq | layers: layers}}
  end

  def train_once(%Sequence{} = seq, x, y_true) do
    {y_pred, seq2} = Sequence.feedforward(seq, x)
    network_error = calc_total_error(y_pred, y_true)
    total_loss_pd = -2 * Enum.sum(network_error)
    ones = Enum.map(network_error, fn _ -> 1.0 end)
    {_, [], seq3} = Sequence.backprop(seq2, total_loss_pd, ones, [])
    seq3
  end

  defp calc_total_error(net_outputs, labels) do
    [labels, net_outputs]
    |> Enum.zip()
    |> Enum.map(fn
      {y_true_item, y_pred_item} -> y_true_item - y_pred_item
    end)
  end
end
