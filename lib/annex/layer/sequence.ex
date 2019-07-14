defmodule Annex.Layer.Sequence do
  @moduledoc """
  The Sequence layer is the container and orchestrator of other layers and is
  used to define a list of Layers that compose a deep neural network.
  """

  alias Annex.{
    Cost,
    Data,
    Data.List1D,
    Data.Shape,
    Defaults,
    Layer,
    Layer.Backprop,
    Layer.Sequence,
    Learner,
    Utils
  }

  require Logger

  @behaviour Learner
  @behaviour Layer

  @type t :: %__MODULE__{
          layers: list(Layer.t()),
          initialized?: boolean(),
          init_options: Keyword.t(),
          train_options: Keyword.t(),
          cost: Cost.t()
        }

  defstruct layers: [],
            initialized?: false,
            init_options: [],
            train_options: [],
            cost: Defaults.get_defaults(:cost)

  @spec build(list(Layer.t()), Keyword.t()) :: Sequence.t()
  def build(layers, opts \\ []) do
    %Sequence{
      layers: layers,
      initialized?: Keyword.get(opts, :initialized?, false)
    }
  end

  @spec get_cost(Sequence.t()) :: Cost.t()
  def get_cost(%Sequence{cost: cost}), do: cost

  @spec get_layers(Sequence.t()) :: list(Layer.t())
  def get_layers(%Sequence{layers: layers}) do
    layers
  end

  defp put_layers(%Sequence{} = seq, layers) do
    %Sequence{seq | layers: layers}
  end

  @impl Learner
  @spec train_opts(keyword()) :: keyword()
  def train_opts(opts) when is_list(opts) do
    Defaults.get_defaults()
    |> Keyword.merge(opts)
    |> Keyword.take([:learning_rate, :cost])
  end

  @impl Learner
  @spec init_learner(t(), any()) :: {:error, any()} | {:ok, t()}
  def init_learner(seq, opts \\ []) do
    init_layer(seq, opts)
  end

  @impl Layer
  @spec data_type :: List1D
  def data_type, do: List1D

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

      Layer.init_layer(layer, layer_opts)
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
        {input_shape, _} = Layer.shapes(layer)
        converted_inputs = Layer.convert(layer, input, input_shape)
        {updated_layer, output} = Layer.feedforward(layer, converted_inputs)
        {output, [updated_layer | layers]}
      end)
      |> case do
        {output, rev_layers} ->
          {output, Enum.reverse(rev_layers)}
      end

    {%Sequence{seq | layers: layers}, output}
  end

  @impl Layer
  @spec backprop(Sequence.t(), any(), keyword()) :: {Sequence.t(), any(), keyword()}
  def backprop(%Sequence{} = seq, seq_losses, backprops) do
    {layers, final_losses, updated_backprop} =
      seq
      |> get_layers()
      |> Enum.reverse()
      |> Enum.reduce({[], seq_losses, backprops}, fn layer, {layers, losses, props} ->
        {_, backprop_shape} = Layer.shapes(layer)
        converted_losses = Layer.convert(layer, losses, backprop_shape)
        {updated_layer, next_losses, next_props} = Layer.backprop(layer, converted_losses, props)
        {[updated_layer | layers], next_losses, next_props}
      end)

    {put_layers(seq, layers), final_losses, updated_backprop}
  end

  @impl Layer
  @spec shapes(t()) :: {Shape.t(), Shape.t()}
  def shapes(%Sequence{} = seq) do
    layers = get_layers(seq)

    {input_shape, _} =
      layers
      |> List.first()
      |> Layer.shapes()

    {_, backprop_shape} =
      layers
      |> List.last()
      |> Layer.shapes()

    {input_shape, backprop_shape}
  end

  @impl Learner
  @spec predict(Sequence.t(), any()) :: any()
  def predict(%Sequence{} = seq, data) do
    {_, prediction} = Layer.feedforward(seq, data)
    prediction
  end

  @impl Learner
  @spec train(t(), any(), any(), Keyword.t()) :: {t(), float()}
  def train(%Sequence{} = seq1, data, labels, _opts) do
    {%Sequence{} = seq2, prediction} = Layer.feedforward(seq1, data)

    prediction_data_type =
      seq1
      |> get_layers()
      |> List.last()
      |> Layer.data_type()

    prediction = Data.to_flat_list(prediction_data_type, prediction)
    labels = Data.to_flat_list(labels)

    error = error(prediction, labels)

    cost = get_cost(seq1)
    # negative gradient so -1.0
    negative_gradient = -1.0 * Cost.derivative(cost, error, data, labels)
    proportioned_error = Utils.proportions(error)

    props = Backprop.new(negative_gradient: negative_gradient)
    {seq3, _next_error, _props} = Layer.backprop(seq2, proportioned_error, props)

    loss = Cost.calculate(cost, error)

    {seq3, loss}
  end

  def error(outputs, labels), do: Utils.subtract(outputs, labels)

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
