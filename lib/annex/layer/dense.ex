defmodule Annex.Layer.Dense do
  @moduledoc """
  `rows` are the number outputs and `columns` are the number of inputs.
  """

  use Annex.Debug, debug: true

  alias Annex.{
    Data,
    Data.DMatrix,
    Data.Shape,
    Layer,
    Layer.Backprop,
    Layer.Dense
  }

  require Data

  import Annex.Utils, only: [is_pos_integer: 1]

  @behaviour Layer

  @type data :: Data.data()

  @type t :: %__MODULE__{
          data_type: Data.type(),
          weights: data() | nil,
          biases: data() | nil,
          rows: pos_integer() | nil,
          columns: pos_integer() | nil,
          input: data() | nil,
          output: data() | nil,
          initialized?: boolean()
        }

  @default_data_type :annex
                     |> Application.get_env(__MODULE__, [])
                     |> Keyword.get(:data_type, DMatrix)

  defstruct weights: nil,
            biases: nil,
            rows: nil,
            columns: nil,
            input: nil,
            output: nil,
            initialized?: false,
            data_type: @default_data_type

  def build(rows, columns) when is_pos_integer(rows) and is_pos_integer(columns) do
    %Dense{
      rows: rows,
      columns: columns
    }
  end

  def build(rows) when is_pos_integer(rows) do
    %Dense{rows: rows}
  end

  def build(rows, columns, weights, biases) do
    debug_assert "Dense rows must be a positive integer", do: is_integer(rows)
    debug_assert "Dense rows must be a positive integer", do: rows > 0
    debug_assert "Dense columns must be a positive integer", do: is_integer(columns)
    debug_assert "Dense columns must be a positive integer", do: columns > 0

    debug_assert "Dense weights must be a list of floats" do
      is_list(weights) || DMatrix.is_type?(weights)
    end

    debug_assert "Dense weights must be a list of floats" do
      Enum.all?(weights, &is_float/1)
    end

    debug_assert "Dense biases must be a list of floats" do
      is_list(biases) || DMatrix.is_type?(biases)
    end

    debug_assert "Dense biases must be a list of floats", do: Enum.all?(biases, &is_float/1)

    debug_assert "Dense biases must be the same length as the count of rows" do
      is_list(biases) && length(biases) == rows
    end

    %Dense{
      biases: DMatrix.build(biases, rows, 1),
      weights: DMatrix.build(weights, rows, columns),
      rows: rows,
      columns: columns,
      initialized?: true
    }
  end

  defp get_output(%Dense{output: o}), do: o

  defp get_weights(%Dense{weights: weights}), do: weights

  defp get_biases(%Dense{biases: biases}), do: biases

  defp get_input(%Dense{input: input}), do: input

  @spec rows(t()) :: pos_integer()
  def rows(%Dense{rows: n}), do: n

  @spec columns(t()) :: pos_integer()
  def columns(%Dense{columns: n}), do: n

  @impl Layer
  @spec init_layer(t(), Keyword.t()) :: {:ok, t()}
  def init_layer(layer, opts \\ [])

  def init_layer(%Dense{initialized?: true} = layer, _opts) do
    {:ok, layer}
  end

  def init_layer(%Dense{initialized?: false} = dense, opts) do
    previous_layer = Keyword.get(opts, :previous_layer)
    next_layer = Keyword.get(opts, :next_layer)
    rows = resolve_init_rows(dense, opts)
    columns = columns(dense)

    debug_assert "rows is a positive integer" do
      is_pos_integer(rows) == true
    end

    debug_assert "columns is a positive integer" do
      is_pos_integer(rows) == true
    end

    initialized = %Dense{
      dense
      | weights: resolve_init_weights(dense, rows, columns),
        biases: resolve_init_biases(dense, rows),
        initialized?: true
    }

    {:ok, initialized}
  end

  defp resolve_init_weights(%Dense{} = dense, rows, columns) do
    dense
    |> get_weights()
    |> build_dmatrix(rows, columns, fn ->
      DMatrix.new_random(rows, columns)
    end)
  end

  defp resolve_init_biases(%Dense{} = dense, rows) do
    dense
    |> get_biases()
    |> build_dmatrix(rows, 1, fn ->
      DMatrix.ones(rows, 1)
    end)
  end

  defp build_dmatrix(item, rows, columns, builder) do
    case item do
      nil ->
        builder.()

      %DMatrix{} = matrix ->
        debug_assert "matrix shape matches rows and columns" do
          shape = DMatrix.shape(matrix)
          shape == {rows, columns}
        end

        matrix

      data when Data.is_flat_data(data) ->
        DMatrix.build(data, rows, columns)
    end
  end

  def resolve_init_rows(%Dense{rows: rows}, _opts) when is_pos_integer(rows) do
    rows
  end

  def resolve_init_rows(%Dense{rows: nil}, opts) do
    prev_layer = Keyword.get(opts, :prev_layer)

    debug_assert "rows must be specified if there is no previous layer" do
      prev_layer != nil
    end

    resolve_init_rows_from_layer(prev_layer)
  end

  defp resolve_init_rows_from_layer(layer) do
    layer
    |> Layer.shape()
    |> case do
      {n} when is_pos_integer(n) -> n
      {n, _} when is_pos_integer(n) -> n
    end
  end

  @impl Layer
  @spec feedforward(t(), DMatrix.t()) :: {t(), DMatrix.t()}
  def feedforward(%Dense{} = dense, inputs) do
    debug_assert "Dense.feedforward/2 input must be dottable with the weights" do
      weights = get_weights(dense)
      {_, dense_columns} = Data.shape(DMatrix, weights)
      {inputs_rows, _} = Data.shape(DMatrix, inputs)
      dense_columns == inputs_rows
    end

    output =
      dense
      |> get_weights()
      |> DMatrix.dot(inputs)
      |> DMatrix.add(get_biases(dense))

    updated_dense = %Dense{
      dense
      | input: inputs,
        output: output
    }

    {updated_dense, output}
  end

  @impl Layer
  @spec backprop(t(), Data.data(), Backprop.t()) :: {t(), Data.data(), Backprop.t()}
  def backprop(%Dense{} = dense, %DMatrix{} = error, props) do
    learning_rate = Backprop.get_learning_rate(props)
    derivative = Backprop.get_derivative(props)
    output = get_output(dense)

    debug_assert "backprop error must have the same shape as output" do
      output_shape = Data.shape(DMatrix, output)
      error_shape = Data.shape(DMatrix, error)
      output_shape == error_shape
    end

    weights = get_weights(dense)
    input = get_input(dense)

    biases = get_biases(dense)

    gradients =
      output
      |> DMatrix.map(derivative)
      |> DMatrix.multiply(error)
      |> DMatrix.multiply(learning_rate)

    input_t = DMatrix.transpose(input)

    debug_assert "gradients must be dottable with input_T" do
      {_, gradients_cols} = Data.shape(DMatrix, gradients)
      {input_rows, _} = Data.shape(DMatrix, input_t)

      gradients_cols == input_rows
    end

    weight_deltas = DMatrix.dot(gradients, input_t)

    updated_weights = DMatrix.subtract(weights, weight_deltas)

    updated_biases = DMatrix.subtract(biases, gradients)

    next_error =
      weights
      |> DMatrix.transpose()
      |> DMatrix.dot(error)

    updated_dense = %Dense{
      dense
      | input: nil,
        output: nil,
        weights: updated_weights,
        biases: updated_biases
    }

    {updated_dense, next_error, props}
  end

  @impl Layer
  @spec data_type(t()) :: Data.type()
  def data_type(%Dense{data_type: data_type}), do: data_type

  @impl Layer
  @spec shape(t()) :: Shape.t()
  def shape(%Dense{} = dense) do
    {rows(dense), columns(dense)}
  end

  defimpl Inspect do
    def inspect(dense, _) do
      shape =
        dense
        |> Dense.shape()
        |> Kernel.inspect()

      "#Dense<[#{shape}]>"
    end
  end
end
