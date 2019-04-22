defmodule Annex.Sequence do
  alias Annex.{Sequence, Layer, Data, Utils, Learner, Cost, Backprop, Activation}
  require Logger

  @behaviour Learner
  @behaviour Layer

  @type t :: %__MODULE__{
          layers: list(Layer.t()),
          initialized?: boolean(),
          init_options: Keyword.t(),
          train_options: Keyword.t(),
          cost_func: (float() -> float())
        }

  defstruct layers: [],
            initialized?: false,
            init_options: [],
            train_options: [],
            cost_func: nil

  def build(opts \\ []) do
    %Sequence{
      layers: Keyword.get(opts, :layers, []),
      initialized?: Keyword.get(opts, :initialized?, false),
      cost_func: Keyword.get(opts, :cost_func, &Cost.mse/1)
    }
  end

  def get_layers(%Sequence{layers: layers}) do
    layers
  end

  defp put_layers(%Sequence{} = seq, layers) do
    %Sequence{seq | layers: layers}
  end

  def train_opts(opts) when is_list(opts) do
    opts
    |> Keyword.take([:learning_rate])
    |> Keyword.put_new(:learning_rate, 0.05)
  end

  @spec init_learner(t(), any()) :: {:error, any()} | {:ok, t()}
  def init_learner(seq, opts \\ []) do
    init_layer(seq, opts)
  end

  def init_layer(seq, opts \\ [])

  def init_layer(%Sequence{initialized?: true} = seq, _opts) do
    {:ok, seq}
  end

  def init_layer(%Sequence{initialized?: false} = seq1, _opts) do
    seq1
    |> get_layers()
    |> chunkify()
    |> Enum.map(fn {previous_layer, layer, next_layer} ->
      layer_opts = [
        previous_layer: previous_layer,
        next_layer: next_layer
      ]

      Layer.init(layer, layer_opts)
    end)
    |> Enum.group_by(
      fn {status, _} -> status end,
      fn {_, initialized} -> initialized end
    )
    |> case do
      %{error: errors} ->
        {:error, errors}

      %{ok: initialized_layers} ->
        initialized_seq = %Sequence{
          seq1
          | layers: initialized_layers,
            initialized?: true
        }

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
    {prediction, _} = Layer.feedforward(seq, data)
    prediction
  end

  @spec backprop(t(), Backprop.t()) :: {t(), Backprop.t()}
  def backprop(%Sequence{} = seq, backprops) do
    {layers, updated_backprop} =
      seq
      |> get_layers()
      |> Enum.reverse()
      |> Enum.reduce({[], backprops}, fn layer, {layers_acc, backprop_acc} ->
        {layer, backprop_acc} = Layer.backprop(layer, backprop_acc)
        {[layer | layers_acc], backprop_acc}
      end)

    {put_layers(seq, layers), updated_backprop}
  end

  @spec train(t(), Data.t(), Data.t(), Keyword.t()) :: {list(float()), t()}
  def train(%Sequence{} = seq, data, labels, _opts) do
    {network_outputs, seq2} = Sequence.feedforward(seq, data)
    outputs = Data.decode(network_outputs)
    labels = Data.decode(labels)

    network_error = calc_network_error(outputs, labels)
    # backprop_error = apply_error_calc(seq, normalized_error)
    # backprop_error =
    backprops = %Backprop{
      net_loss: calculate_network_error_pd(network_error),
      loss_pds: Utils.proportions(network_error),
      learning_rate: 0.05,
      derivative: &Activation.sigmoid_deriv/1
    }

    {seq3, _backprop} = Sequence.backprop(seq2, backprops)
    cost_func = get_cost_func(seq)

    loss =
      labels
      |> Utils.zipmap(outputs, fn ax, bx -> ax - bx end)
      |> cost_func.()

    {loss, seq3}
  end

  def total_loss_pd(outputs, labels) do
    outputs
    |> calc_network_error(labels)
    |> calculate_network_error_pd()
  end

  def encoder, do: Annex.Data

  defp calculate_network_error_pd(network_error) do
    -2 * Enum.sum(network_error)
  end

  defp calc_network_error(net_outputs, labels) do
    labels
    |> Utils.zip(net_outputs)
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

  defp get_cost_func(%Sequence{cost_func: cost_func}), do: cost_func
end
