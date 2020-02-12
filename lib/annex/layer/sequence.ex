defmodule Annex.Layer.Sequence do
  @moduledoc """
  The Sequence layer is the container and orchestrator of other layers and is
  used to define a list of Layers that compose a deep neural network.
  """

  alias Annex.{
    AnnexError,
    Data,
    Data.DMatrix,
    Layer,
    Layer.Backprop,
    Layer.Sequence,
    Learner,
    Shape
  }

  require Logger

  @behaviour Learner
  use Layer

  @type layers :: MapArray.t()

  @type t :: %__MODULE__{
          layers: layers,
          initialized?: boolean(),
          init_options: Keyword.t(),
          train_options: Keyword.t()
        }

  defstruct layers: %{},
            layer_configs: [],
            initialized?: false,
            init_options: [],
            train_options: []

  @impl Layer
  @spec init_layer(LayerConfig.t(Sequence)) :: t()
  def init_layer(%LayerConfig{} = cfg) do
    with(
      {:ok, :layers, layer_configs} <- LayerConfig.fetch(cfg, :layers),
      layers <- do_init_layers(layer_configs)
    ) do
      %Sequence{layers: layers, layer_configs: layer_configs, initialized?: true}
    else
      {:error, :layers, %AnnexError{} = err} ->
        raise err
    end
  end

  defp do_init_layers(layer_configs) do
    layer_configs
    |> Enum.reduce([], fn %LayerConfig{} = layer_config, acc ->
      [LayerConfig.init_layer(layer_config) | acc]
    end)
    |> Enum.reverse()
    |> MapArray.new()
  end

  @spec get_layers(t()) :: layers
  def get_layers(%Sequence{layers: layers}), do: layers

  @impl Learner
  @spec init_learner(t() | LayerConfig.t(Sequence), Keyword.t()) :: t() | no_return()
  def init_learner(seq, opts \\ [])

  def init_learner(%Sequence{initialized?: true} = seq, opts), do: seq

  def init_learner(%Sequence{layer_configs: layer_configs, initialized?: false}, opts) do
    Sequence
    |> LayerConfig.build(layers: layer_configs)
    |> init_learner(opts)
  end

  def init_learner(%LayerConfig{} = cfg, _opts) do
    init_layer(cfg)
  end

  @impl Layer
  @spec data_type(t()) :: DMatrix
  def data_type(_), do: DMatrix

  @impl Layer
  @spec feedforward(Sequence.t(), Data.data()) :: {Sequence.t(), Data.data()}
  def feedforward(%Sequence{} = seq, seq_inputs) do
    layers1 = get_layers(seq)

    {output, layers2} =
      MapArray.reduce(layers1, {seq_inputs, %{}}, fn layer1, {inputs1, layers_acc}, i ->
        inputs2 = do_convert_for_feedforward(layers1, i, inputs1)
        {layer2, output} = Layer.feedforward(layer1, inputs2)

        {output, MapArray.append(layers_acc, layer2)}
      end)

    {%Sequence{seq | layers: layers2}, output}
  end

  defp do_convert_for_feedforward(layers, start_index, data) do
    layers
    |> MapArray.seek_up(start_index, fn layer ->
      Layer.has_shapes?(layer) && Layer.data_type(layer)
    end)
    |> case do
      :error ->
        data

      {:ok, layer} ->
        data_type = Layer.data_type(layer)
        shape = shape_for_feedforward(layer)
        Data.convert(data_type, data, shape)
    end
  end

  defp do_convert_for_backprop(layers, start_index, data) do
    layers
    |> MapArray.seek_down(start_index, fn layer ->
      Layer.has_shapes?(layer) && Layer.data_type(layer)
    end)
    |> case do
      :error ->
        data

      {:ok, layer} ->
        data_type = Layer.data_type(layer)
        shape = shape_for_backprop(layer)
        Data.convert(data_type, data, shape)
    end
  end

  defp shape_for_feedforward(layer) do
    case Layer.input_shape(layer) do
      [_rows, columns] -> [columns, :any]
      [columns] -> [columns, :any]
    end
  end

  defp shape_for_backprop(layer) do
    case Layer.output_shape(layer) do
      [_columns, rows] -> [rows, :any]
      [_] -> [1, :any]
    end
  end

  @impl Layer
  @spec backprop(t(), Data.data(), Backprop.t()) :: {t(), Data.data(), Backprop.t()}
  def backprop(%Sequence{} = seq, seq_errors, seq_backprops) do
    layers = get_layers(seq)

    {
      output_errors,
      output_props,
      output_layers
    } =
      MapArray.reverse_reduce(layers, {seq_errors, seq_backprops, %{}}, fn
        layer, {errors, backprops, layers_acc}, i ->
          errors2 = do_convert_for_backprop(layers, i, errors)
          {layer2, errors3, backprops2} = Layer.backprop(layer, errors2, backprops)
          layers_acc2 = Map.put(layers_acc, i, layer2)
          {errors3, backprops2, layers_acc2}
      end)

    {%Sequence{seq | layers: output_layers}, output_errors, output_props}
  end

  @impl Layer
  @spec shapes(t()) :: {Shape.t(), Shape.t()}
  def shapes(%Sequence{} = seq) do
    {input_shape, _} = first_shape(seq)
    {_, output_shape} = last_shape(seq)
    {input_shape, output_shape}
  end

  defp first_shape(%Sequence{} = seq) do
    seq
    |> get_layers
    |> MapArray.seek_up(fn layer ->
      Layer.has_shapes?(layer)
    end)
    |> case do
      :error ->
        raise Annex.AnnexError,
          message: """
          Sequence requires at least one shaped layer.
          """

      {:ok, layer} ->
        Layer.shapes(layer)
    end
  end

  defp last_shape(%Sequence{} = seq) do
    # error case cover in first_shape.
    {:ok, layer} =
      seq
      |> get_layers
      |> MapArray.seek_down(fn layer ->
        Layer.has_shapes?(layer)
      end)

    Layer.shapes(layer)
  end

  @impl Learner
  @spec predict(Sequence.t(), any()) :: Data.data()
  def predict(%Sequence{} = seq, data) do
    {_, prediction} = Layer.feedforward(seq, data)
    prediction
  end

  defimpl Inspect do
    def inspect(seq, _) do
      details =
        seq
        |> Sequence.get_layers()
        |> MapArray.map(fn %module{} = layer ->
          Kernel.inspect({module, data_type(layer), shapes(layer)})
        end)
        |> Enum.intersperse("\n\t")
        |> IO.iodata_to_binary()

      "#Sequence<[\n\t#{details}\n]>"
    end

    def data_type(layer) do
      if Layer.has_data_type?(layer) do
        Layer.data_type(layer)
      end
    end

    def shapes(layer) do
      if Layer.has_shapes?(layer) do
        Layer.shapes(layer)
      end
    end
  end
end
