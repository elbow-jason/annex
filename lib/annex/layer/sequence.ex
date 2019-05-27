defmodule Annex.Layer.Sequence do
  alias Annex.{
    Cost,
    Layer,
    Layer.Backprop,
    Layer.ListLayer,
    Layer.Sequence,
    Learner,
    Utils
  }

  require Logger

  @behaviour Learner
  @behaviour Layer

  use ListLayer

  @type t :: %__MODULE__{
          layers: list(Layer.t()),
          initialized?: boolean(),
          init_options: Keyword.t(),
          train_options: Keyword.t()
        }

  defstruct layers: [],
            initialized?: false,
            init_options: [],
            train_options: []

  def build(opts \\ []) do
    %Sequence{
      layers: Keyword.get(opts, :layers, []),
      initialized?: Keyword.get(opts, :initialized?, false)
    }
  end

  def get_layers(%Sequence{layers: layers}) do
    layers
  end

  defp put_layers(%Sequence{} = seq, layers) do
    %Sequence{seq | layers: layers}
  end

  @impl Learner
  @spec train_opts(keyword()) :: keyword()
  def train_opts(opts) when is_list(opts) do
    opts
    |> Keyword.take([:learning_rate])
    |> Keyword.put_new(:learning_rate, 0.05)
  end

  @impl Learner
  @spec init_learner(t(), any()) :: {:error, any()} | {:ok, t()}
  def init_learner(seq, opts \\ []) do
    init_layer(seq, opts)
  end

  @impl Layer
  @spec init_layer(Sequence.t(), any()) :: {:error, any()} | {:ok, Sequence.t()}
  def init_layer(seq, opts \\ [])

  def init_layer(%Sequence{initialized?: true} = seq, _opts) do
    {:ok, seq}
  end

  def init_layer(%Sequence{initialized?: false} = seq1, _opts) do
    seq1
    |> get_layers()
    |> prepare_for_chunkify()
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

  @impl Layer
  @spec feedforward(Sequence.t(), any()) :: {Sequence.t(), any()}
  def feedforward(%Sequence{} = seq, inputs) do
    {output, layers} =
      seq
      |> get_layers()
      |> Enum.reduce({inputs, []}, fn layer, {input, layers} ->
        prev_layer = List.first(layers) || seq
        encoded_inputs = Layer.convert(input, prev_layer, layer)
        {updated_layer, output} = Layer.feedforward(layer, encoded_inputs)
        {output, [updated_layer | layers]}
      end)
      |> case do
        {output, rev_layers} ->
          {output, Enum.reverse(rev_layers)}
      end

    {%Sequence{seq | layers: layers}, output}
  end

  @impl Learner
  @spec predict(Sequence.t(), any()) :: any()
  def predict(%Sequence{} = seq, data) do
    {_, prediction} = Layer.feedforward(seq, data)
    prediction
  end

  @impl Layer
  @spec backprop(Sequence.t(), any(), keyword()) :: {Sequence.t(), any(), keyword()}
  def backprop(%Sequence{} = seq, seq_losses, backprops) do
    {layers, final_losses, updated_backprop} =
      seq
      |> get_layers()
      |> Enum.reverse()
      |> Enum.reduce({[], seq_losses, backprops}, fn layer, {layers, losses, props} ->
        prev_layer = List.first(layers) || seq
        encoded_losses = Layer.convert(losses, prev_layer, layer)
        {updated_layer, next_losses, next_props} = Layer.backprop(layer, encoded_losses, props)
        {[updated_layer | layers], next_losses, next_props}
      end)

    {put_layers(seq, layers), final_losses, updated_backprop}
  end

  @impl Learner
  @spec train(t(), ListLayer.t(), ListLayer.t(), Keyword.t()) :: {t(), ListLayer.t()}
  def train(%Sequence{} = seq1, data, labels, opts) do
    cost = Keyword.get(opts, :cost, Cost.MeanSquaredError)
    cost_func = &cost.calculate/2

    {%Sequence{} = seq2, prediction} = Layer.feedforward(seq1, data)

    last_layer =
      seq2
      |> get_layers()
      |> List.last()

    labels = Layer.convert(labels, seq2, seq2)
    prediction = Layer.convert(prediction, last_layer, seq2)

    network_error = calc_network_error(prediction, labels)
    network_error_pd = calculate_network_error_pd(network_error)
    loss_pds = Utils.proportions(network_error)

    props = Backprop.new(net_loss: network_error_pd, cost_func: cost_func)

    {seq3, _next_loss_pds, _props} = Layer.backprop(seq2, loss_pds, props)

    loss = cost_func.(labels, prediction)

    {seq3, loss}
  end

  def total_loss_pd(outputs, labels) do
    outputs
    |> calc_network_error(labels)
    |> calculate_network_error_pd()
  end

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

  defp prepare_for_chunkify(layers) do
    [nil] ++ layers ++ [nil]
  end

  defp chunkify(prepared_layers) do
    chunkify(prepared_layers, [])
  end

  defp chunkify([prev, current, next], acc) do
    Enum.reverse([{prev, current, next} | acc])
  end

  defp chunkify([prev, current, next | rest], acc) do
    chunkify([current, next | rest], [{prev, current, next} | acc])
  end
end
