defmodule Annex.Layer.Dense do
  @moduledoc """
  `rows` are the number outputs and `columns` are the number of inputs.
  """

  use Annex.Debug, debug: true

  alias Annex.{
    # Data.List1D,
    Data.Shape,
    Data.DMatrix,
    Layer,
    Layer.Backprop,
    Layer.Dense
    # Layer.Neuron,
    # Utils
  }

  import Annex.Utils, only: [is_pos_integer: 1]

  @behaviour Layer

  @type t :: %__MODULE__{
          weights: DMatrix.t(),
          biases: DMatrix.t(),
          rows: pos_integer(),
          columns: pos_integer() | nil,
          input: list(float()),
          output: list(float()),
          initialized?: boolean()
        }

  defstruct weights: nil,
            biases: nil,
            rows: nil,
            columns: nil,
            input: [],
            output: [],
            initialized?: false

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
      length(biases) == rows
    end

    %Dense{
      biases: DMatrix.build(biases, rows, 1),
      weights: DMatrix.build(weights, rows, columns),
      rows: rows,
      columns: columns,
      initialized?: true
    }
  end

  defp put_output(%Dense{} = dense, output) do
    %Dense{dense | output: output}
  end

  defp get_output(%Dense{output: o}), do: o

  defp get_weights(%Dense{weights: weights}), do: IO.inspect(weights, label: :GET_WEEEEIGHTS)

  defp get_biases(%Dense{biases: biases}), do: biases

  defp put_input(%Dense{} = dense, input) do
    %Dense{dense | input: input}
  end

  @spec rows(t()) :: pos_integer()
  def rows(%Dense{rows: n}), do: n

  @spec columns(t()) :: pos_integer()
  def columns(%Dense{columns: n}), do: n

  @impl Layer
  @spec init_layer(t(), Keyword.t()) :: {:ok, t()}
  def init_layer(layer, opts \\ [])

  def init_layer(%Dense{initialized?: false} = layer, _opts) do
    rows = rows(layer)
    columns = columns(layer)

    initialized = %Dense{
      layer
      | weights: DMatrix.new_random(rows, columns),
        biases: DMatrix.ones(rows, 1),
        initialized?: true
    }

    {:ok, initialized}
  end

  def init_layer(%Dense{initialized?: true} = layer, _opts) do
    {:ok, layer}
  end

  @impl Layer
  @spec feedforward(t(), DMatrix.t()) :: {t(), DMatrix.t()}
  def feedforward(%Dense{} = layer, input) do
    output =
      layer
      |> get_weights()
      |> DMatrix.dot(input)
      |> DMatrix.add(get_biases(layer))

    updated_layer =
      layer
      |> put_input(input)
      |> put_output(output)

    {updated_layer, output}
  end

  @impl Layer
  # @spec backprop(t(), DMatrix.t(), Backprop.t()) :: {t(), DMatrix.t(), Backprop.t()}
  @spec backprop(t(), DMatrix.t(), keyword) :: {t(), DMatrix.t(), keyword}
  def backprop(%Dense{} = dense, error, props) do
    learning_rate = Backprop.get_learning_rate(props)
    derivative = Backprop.get_derivative(props)
    # negative_gradient = Backprop.get_negative_gradient(props)

    output = get_output(dense)

    weights = get_weights(dense)
    biases = get_biases(dense)

    weights_t = DMatrix.transpose(weights)

    adjusted_error = DMatrix.multiply(error, learning_rate)

    gradients =
      output
      |> DMatrix.map(derivative)
      |> DMatrix.dot(adjusted_error)

    weight_deltas = DMatrix.dot(weights_t, gradients)

    IO.inspect(weights, label: :weights_asdasdasd)
    IO.inspect(weight_deltas, label: :weight_deltas_asdasdasd)

    updated_weights = DMatrix.add(weights, weight_deltas)
    updated_biases = DMatrix.add(biases, gradients)
    next_error = DMatrix.multiply(weights_t, error)

    updated_dense = %Dense{
      dense
      | input: nil,
        output: nil,
        weights: updated_weights,
        biases: updated_biases
    }

    # {layer, next_error, props}
    {updated_dense, next_error, props}
  end

  @impl Layer
  @spec data_type :: DMatrix
  def data_type, do: DMatrix

  @impl Layer
  @spec shapes(t()) :: {Shape.t(), Shape.t()}
  def shapes(%Dense{} = dense) do
    {{columns(dense)}, {rows(dense)}}
  end

  defp put_random_weights(%Dense{rows: rows, columns: columns} = layer) do
  end
end
