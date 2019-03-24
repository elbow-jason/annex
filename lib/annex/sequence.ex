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

  def initialize(seq, opts \\ [])

  def initialize(%Sequence{initialized?: true} = seq, _opts) do
    {:ok, seq}
  end

  def initialize(%Sequence{initialized?: false} = seq1, _opts) do
    seq1
    |> get_layers()
    |> chunkify()
    |> Enum.map(fn {previous_layer, layer, next_layer} ->
      layer_opts = [
        previous_layer: previous_layer,
        next_layer: next_layer
      ]

      Layer.initialize(layer, layer_opts)
    end)
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

  def train_once(%Sequence{} = seq, data, labels) do
    {outputs, seq2} = Sequence.feedforward(seq, data)

    total_loss_pd = total_loss_pd(outputs, labels)

    ones = Enum.map(labels, fn _ -> 1.0 end)
    {_, [], seq3} = Sequence.backprop(seq2, total_loss_pd, ones, [])
    seq3
  end

  def total_loss_pd(outputs, labels) do
    outputs
    |> calc_network_error(labels)
    |> calculate_network_error_pd()
  end

  def encoder, do: Annex.Data

  def calculate_network_error_pd(network_error) do
    -2 * Enum.sum(network_error)
  end

  def calc_network_error(net_outputs, labels) do
    [labels, net_outputs]
    |> Enum.zip()
    |> Enum.map(fn
      {y_true_item, y_pred_item} -> y_true_item - y_pred_item
    end)
  end

  defp chunkify([]) do
    []
  end

  defp chunkify([one]) do
    [{nil, one, nil}]
  end

  defp chunkify([one, two]) do
    [{one, two, nil}, {nil, one, two}]
  end

  defp chunkify([prev, current, next | rest]) do
    acc = [
      {prev, current, next},
      {nil, prev, current}
    ]

    chunkify([current, next | rest], acc)
  end

  defp chunkify([prev, current, next | rest], acc) do
    chunkify([current, next | rest], [{prev, current, next} | acc])
  end

  defp chunkify([prev, current], acc) do
    Enum.reverse([{prev, current, nil} | acc])
  end
end
