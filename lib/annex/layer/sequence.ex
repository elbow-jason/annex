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
    Data.Shape,
    Defaults,
    Layer,
    Layer.Backprop,
    Layer.Sequence,
    Learner
  }

  require Logger

  @behaviour Learner
  @behaviour Layer

  @type layers :: MapArray.t()

  @type t :: %__MODULE__{
          layers: layers,
          initialized?: boolean(),
          init_options: Keyword.t(),
          train_options: Keyword.t(),
          cost: Cost.t()
        }

  defstruct layers: %{},
            initialized?: false,
            init_options: [],
            train_options: [],
            cost: Defaults.get_defaults(:cost)

  @spec build(list(Layer.t()), Keyword.t()) :: Sequence.t()
  def build(layers, _opts \\ []) when is_list(layers) do
    %Sequence{
      initialized?: false,
      layers: MapArray.new(layers)
    }
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
  @spec init_learner(t(), any()) :: {:error, any()} | {:ok, t()}
  def init_learner(seq, opts \\ []) do
    init_layer(seq, opts)
  end

  @impl Layer
  @spec data_type(t()) :: DMatrix
  def data_type(_), do: DMatrix

  @impl Layer
  @spec init_layer(Sequence.t(), any()) :: {:error, any()} | {:ok, Sequence.t()}
  def init_layer(seq, opts \\ [])

  def init_layer(%Sequence{initialized?: true} = seq, _opts) do
    {:ok, seq}
  end

  def init_layer(%Sequence{initialized?: false} = seq1, _opts) do
    initialized_layers =
      seq1
      |> get_layers()
      |> MapArray.map(fn layer, i ->
        case Layer.init_layer(layer, []) do
          {:ok, layer} ->
            {i, layer}

          err ->
            raise Annex.AnnexError,
              message: """
              Annex.Layer.Sequence failed to initialize layer.

              error: #{inspect(err)}
              layer: #{inspect(layer)}
              sequence: #{inspect(seq1)}
              """
        end
      end)
      |> Map.new()

    initialized_seq = %Sequence{
      seq1
      | layers: initialized_layers,
        initialized?: true
    }

    {:ok, initialized_seq}
  end

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
      Layer.shape(layer) && Layer.data_type(layer)
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
      Layer.shape(layer) && Layer.data_type(layer)
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
    case Layer.shape(layer) do
      {_rows, columns} -> {columns, :any}
      {columns} -> {columns, :any}
    end
  end

  defp shape_for_backprop(layer) do
    case Layer.shape(layer) do
      {rows, _columns} -> {rows, :any}
      {_} -> {1, :any}
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
  @spec shape(t()) :: Shape.t()
  def shape(%Sequence{} = seq) do
    {_rows, columns} = first_shape(seq)
    {rows, _columns} = last_shape(seq)
    {rows, columns}
  end

  defp first_shape(%Sequence{} = seq) do
    seq
    |> get_layers
    |> MapArray.seek_up(fn layer ->
      Layer.shape(layer)
    end)
    |> case do
      :error ->
        raise Annex.AnnexError,
          message: """
          Sequence requires at least one shaped layer.
          """

      {:ok, layer} ->
        Layer.shape(layer)
    end
  end

  defp last_shape(%Sequence{} = seq) do
    seq
    |> get_layers
    |> MapArray.seek_down(fn layer ->
      Layer.shape(layer)
    end)
    |> case do
      :error ->
        raise Annex.AnnexError,
          message: """
          Sequence requires at least one shaped layer.
          """

      {:ok, layer} ->
        Layer.shape(layer)
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
          Kernel.inspect({module, Layer.data_type(layer), Layer.shape(layer)})
        end)
        |> Enum.intersperse("\n\t")
        |> IO.iodata_to_binary()

      "#Sequence<[\n\t#{details}\n]>"
    end
  end
end
