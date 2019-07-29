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
    Layer.Dense,
    Utils
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

  @spec build(pos_integer, pos_integer) :: t()
  def build(rows, columns) when is_pos_integer(rows) and is_pos_integer(columns) do
    %Dense{
      rows: rows,
      columns: columns
    }
  end

  @spec build(pos_integer()) :: t()
  def build(rows) when is_pos_integer(rows) do
    %Dense{rows: rows}
  end

  @spec build(pos_integer, pos_integer, [float, ...], [float, ...]) :: t()
  def build(rows, columns, weights, biases, opts \\ []) do
    debug_assert "Dense rows must be a positive integer" do
      is_int? = is_integer(rows)
      is_positive? = rows > 0
      is_int? && is_positive?
    end

    debug_assert "Dense columns must be a positive integer" do
      is_int? = is_integer(columns)
      is_positive? = columns > 0
      is_int? && is_positive?
    end

    debug_assert "Dense weights must be an Annex.Data" do
      type = Data.infer_type(weights)
      Data.is_type?(type, weights)
    end

    debug_assert "Dense biases must be a list of floats" do
      type = Data.infer_type(biases)
      Data.is_type?(type, biases)
    end

    debug_assert "Dense biases shape must be compatible with Dense shape" do
      {biases_rows, biases_columns} =
        case Data.shape(biases) do
          {rows} -> {rows, 1}
          {rows, cols} -> {rows, cols}
        end

      biases_rows == rows && biases_columns == 1
    end

    data_type = Keyword.get(opts, :data_type, @default_data_type)

    %Dense{
      biases: Data.cast!(data_type, biases, {rows, 1}),
      weights: Data.cast!(data_type, weights, {rows, columns}),
      rows: rows,
      columns: columns,
      initialized?: true,
      data_type: data_type
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
    |> data_type()
    |> build_data(get_weights(dense), rows, columns, fn ->
      Utils.random_weights(rows * columns)
    end)
  end

  defp resolve_init_biases(%Dense{} = dense, rows) do
    dense
    |> data_type()
    |> build_data(get_biases(dense), rows, 1, fn ->
      fn -> 1.0 end
      |> Stream.repeatedly()
      |> Enum.take(rows)
    end)
  end

  defp build_data(type, data, rows, columns, builder) when is_atom(type) do
    data = data || builder.()
    Data.cast!(type, data, {rows, columns})
  end

  defp resolve_init_rows(%Dense{rows: rows}, _opts) when is_pos_integer(rows) do
    rows
  end

  defp resolve_init_rows(%Dense{rows: nil}, opts) do
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
  @spec feedforward(t(), Data.data()) :: {t(), Data.data()}
  def feedforward(%Dense{} = dense, inputs) do
    debug_assert "Dense.feedforward/2 input must be dottable with the weights" do
      weights = get_weights(dense)
      {_, dense_columns} = Data.shape(weights)
      {inputs_rows, _} = Data.shape(inputs)
      dense_columns == inputs_rows
    end

    biases = get_biases(dense)

    output =
      dense
      |> get_weights()
      |> Data.apply_op(:dot, [inputs])
      |> Data.apply_op(:add, [biases])

    updated_dense = %Dense{
      dense
      | input: inputs,
        output: output
    }

    {updated_dense, output}
  end

  @impl Layer
  @spec backprop(t(), Data.data(), Backprop.t()) :: {t(), Data.data(), Backprop.t()}
  def backprop(%Dense{} = dense, error, props) do
    learning_rate = Backprop.get_learning_rate(props)
    derivative = Backprop.get_derivative(props)
    output = get_output(dense)

    debug_assert "backprop error must have the same shape as output" do
      output_shape = Data.shape(output)
      error_shape = Data.shape(error)
      output_shape == error_shape
    end

    weights = get_weights(dense)
    input = get_input(dense)

    biases = get_biases(dense)

    gradients =
      output
      |> Data.apply_op(:map, [derivative])
      |> Data.apply_op(:multiply, [error])
      |> Data.apply_op(:multiply, [learning_rate])

    input_t = Data.apply_op(input, :transpose, [])

    debug_assert "gradients must be dottable with input_T" do
      {_, gradients_cols} = Data.shape(gradients)
      {input_rows, _} = Data.shape(input_t)

      gradients_cols == input_rows
    end

    weight_deltas = Data.apply_op(gradients, :dot, [input_t])

    updated_weights = Data.apply_op(weights, :subtract, [weight_deltas])

    updated_biases = Data.apply_op(biases, :subtract, [gradients])

    next_error =
      weights
      |> Data.apply_op(:transpose, [])
      |> Data.apply_op(:dot, [error])

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
