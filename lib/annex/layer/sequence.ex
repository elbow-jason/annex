defmodule Annex.Layer.Sequence do
  @moduledoc """
  The Sequence layer is the container and orchestrator of other layers and is
  used to define a list of Layers that compose a deep neural network.
  """

  alias Annex.{
    AnnexError,
    Cost,
    Data,
    Data.DMatrix,
    Defaults,
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
          train_options: Keyword.t(),
          cost: Cost.t()
        }

  defstruct layers: %{},
            layer_configs: [],
            initialized?: false,
            init_options: [],
            train_options: [],
            cost: Defaults.get_defaults(:cost)

  @impl Layer
  @spec init_layer(LayerConfig.t(Sequence)) :: {:ok, t()} | {:error, AnnexError.t()}
  def init_layer(%LayerConfig{} = cfg) do
    with(
      {:ok, :layers, layer_configs} <- LayerConfig.fetch(cfg, :layers),
      {:ok, layers} <- do_init_layers(layer_configs)
    ) do
      {:ok, %Sequence{layers: layers, layer_configs: layer_configs}}
    else
      {:error, :layers, %AnnexError{} = err} ->
        {:error, err}

      {:error, %AnnexError{}} = err ->
        err
    end
  end

  defp do_init_layers(layer_configs) do
    layer_configs
    |> Enum.reduce_while([], fn %LayerConfig{} = layer_config, acc ->
      case LayerConfig.init_layer(layer_config) do
        {:ok, built_layer} ->
          {:cont, [built_layer | acc]}

        {:error, _} = error ->
          {:halt, error}
      end
    end)
    |> case do
      rev_built_layers when is_list(rev_built_layers) ->
        built_layers =
          rev_built_layers
          |> Enum.reverse()
          |> MapArray.new()

        {:ok, built_layers}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @spec add_layer(t(), struct()) :: t()
  def add_layer(%Sequence{} = seq, %_{} = layer) do
    %Sequence{seq | layers: seq |> get_layers() |> MapArray.append(layer)}
  end

  @spec get_cost(t()) :: Cost.t()
  def get_cost(%Sequence{cost: cost}), do: cost

  @spec get_layers(t()) :: layers
  def get_layers(%Sequence{layers: layers}), do: layers

  @impl Learner
  @spec train_opts(keyword()) :: keyword()
  def train_opts(opts) when is_list(opts) do
    Defaults.get_defaults()
    |> Keyword.merge(opts)
    |> Keyword.take([:learning_rate, :cost])
  end

  @impl Learner
  @spec init_learner(t() | LayerConfig.t(Sequence), Keyword.t()) :: {:error, any()} | {:ok, t()}
  def init_learner(seq, opts \\ [])

  def init_learner(%Sequence{layer_configs: layer_configs}, opts) do
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
        with(
          {:ok, inputs2} <- do_convert_for_feedforward(layers1, i, inputs1),
          {layer2, output} <- Layer.feedforward(layer1, inputs2)
        ) do
          {output, MapArray.append(layers_acc, layer2)}
        else
          {:error, %AnnexError{} = error} ->
            details = [
              step: :backprop,
              layer: layer1,
              index: i,
              sequence: seq
            ]

            raise AnnexError.add_details(error, details)
        end
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
        {:ok, data}

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
        {:ok, data}

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
  @spec backprop(Sequence.t(), Data.data(), Backprop.t()) ::
          {Sequence.t(), Data.data(), Backprop.t()}
  def backprop(%Sequence{} = seq, seq_errors, seq_backprops) do
    layers = get_layers(seq)

    {
      output_errors,
      output_props,
      output_layers
    } =
      MapArray.reverse_reduce(layers, {seq_errors, seq_backprops, %{}}, fn
        layer, {errors, backprops, layers_acc}, i ->
          with(
            {:ok, errors2} <- do_convert_for_backprop(layers, i, errors),
            {layer2, errors3, backprops2} <- Layer.backprop(layer, errors2, backprops),
            layers_acc2 <- Map.put(layers_acc, i, layer2)
          ) do
            {errors3, backprops2, layers_acc2}
          else
            {:error, %AnnexError{} = error} ->
              details = [
                step: :backprop,
                layer: layer,
                index: i,
                sequence: seq
              ]

              raise AnnexError.add_details(error, details)
          end
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
    seq
    |> get_layers
    |> MapArray.seek_down(fn layer ->
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

  @impl Learner
  @spec predict(Sequence.t(), any()) :: Data.data()
  def predict(%Sequence{} = seq, data) do
    {_, prediction} = Layer.feedforward(seq, data)
    prediction
  end

  @impl Learner
  @spec train(t(), Data.data(), Data.data(), Keyword.t()) :: {t(), Learner.train_output()}
  def train(%Sequence{} = seq1, data, labels, _opts) do
    {%Sequence{} = seq2, prediction} = Layer.feedforward(seq1, data)

    prediction = Data.to_flat_list(prediction)
    labels = Data.to_flat_list(labels)

    errors =
      prediction
      |> error(labels)
      |> Data.to_flat_list()

    props = Backprop.new()
    {seq3, _error2, _props} = Layer.backprop(seq2, errors, props)

    loss =
      seq1
      |> get_cost()
      |> Cost.calculate(errors)

    output = %{
      loss: loss,
      data: data,
      labels: labels,
      prediction: prediction,
      errors: errors
    }

    {seq3, output}
  end

  @spec error(Data.data(), Data.data()) :: Data.data()
  def error(outputs, labels) do
    Data.apply_op(outputs, :subtract, [labels])
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
