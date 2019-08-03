defmodule Annex.Layer.Dense do
  @moduledoc """
  `rows` are the number outputs and `columns` are the number of inputs.
  """

  use Annex.Debug, debug: true

  alias Annex.{
    Data,
    Data.DMatrix,
    Data.Shape,
    Layer.Backprop,
    Layer.Dense,
    LayerConfig,
    Utils
  }

  require Data

  use Annex.Layer

  @type data :: Data.data()

  @type t :: %__MODULE__{
          data_type: Data.type(),
          weights: data() | nil,
          biases: data() | nil,
          rows: pos_integer() | nil,
          columns: pos_integer() | nil,
          input: data() | nil,
          output: data() | nil
        }

  defp default_data_type do
    :annex
    |> Application.get_env(__MODULE__, [])
    |> Keyword.get(:data_type, DMatrix)
  end

  defstruct weights: nil,
            biases: nil,
            rows: nil,
            columns: nil,
            input: nil,
            output: nil,
            data_type: nil

  @impl Layer
  @spec init_layer(LayerConfig.t()) :: {:ok, t()} | {:error, AnnexError.t()}
  def init_layer(%LayerConfig{} = cfg) do
    with(
      {:ok, :data_type, data_type} <- build_data_type(cfg),
      {:ok, :rows, rows} <- build_rows(cfg),
      {:ok, :columns, columns} <- build_columns(cfg),
      {:ok, :weights, weights} <- build_weights(cfg, data_type, rows, columns),
      {:ok, :biases, biases} <- build_biases(cfg, data_type, rows)
    ) do
      dense = %Dense{
        biases: biases,
        weights: weights,
        rows: rows,
        columns: columns,
        data_type: data_type
      }

      {:ok, dense}
    else
      {:error, _field, error} ->
        {:error, error}
    end
  end

  defp build_rows(cfg) do
    with(
      {:ok, :rows, rows} <- LayerConfig.fetch(cfg, :rows),
      :ok <-
        validate :rows, "must be a positive integer" do
          is_int? = is_integer(rows)
          is_positive? = rows > 0
          is_int? && is_positive?
        end
    ) do
      {:ok, :rows, rows}
    end
  end

  defp build_columns(cfg) do
    with(
      {:ok, :columns, columns} <- LayerConfig.fetch(cfg, :columns),
      :ok <-
        validate :columns, "must be a positive integer" do
          is_int? = is_integer(columns)
          is_positive? = columns > 0
          is_int? && is_positive?
        end
    ) do
      {:ok, :columns, columns}
    end
  end

  defp build_data_type(cfg) do
    with(
      {:ok, :data_type, data_type} <-
        LayerConfig.fetch_lazy(cfg, :data_type, fn ->
          default_data_type()
        end),
      :ok <-
        validate :data_type, "must be a module" do
          Utils.is_module?(data_type)
        end
    ) do
      {:ok, :data_type, data_type}
    end
  end

  defp build_weights(cfg, data_type, rows, columns) do
    with(
      {:ok, :weights, weights} <-
        LayerConfig.fetch_lazy(cfg, :weights, fn ->
          Utils.random_weights(rows * columns)
        end),
      :ok <-
        validate :weights, "must be an Annex.Data" do
          type = Data.infer_type(weights)
          Data.is_type?(type, weights)
        end,
      :ok <-
        validate :weights, "data size must be the same as the layer" do
          {weights_rows, weights_columns} =
            case Data.shape(weights) do
              {w_rows} -> {w_rows, 1}
              {w_rows, w_cols} -> {w_rows, w_cols}
            end

          weights_size = weights_rows * weights_columns
          layer_size = rows * columns
          weights_size == layer_size
        end,
      {:ok, casted_weights} <- Data.cast(data_type, weights, {rows, columns})
    ) do
      {:ok, :weights, casted_weights}
    end
  end

  defp build_biases(cfg, data_type, rows) do
    with(
      {:ok, :biases, biases} <-
        LayerConfig.fetch_lazy(cfg, :biases, fn ->
          Utils.ones(rows)
        end),
      :ok <-
        validate :biases, "must be an Annex.Data" do
          type = Data.infer_type(biases)
          Data.is_type?(type, biases)
        end,
      :ok <-
        validate :biases, "data size must match rows" do
          {biases_rows, biases_columns} =
            case Data.shape(biases) do
              {b_rows} -> {b_rows, 1}
              {b_rows, b_cols} -> {b_rows, b_cols}
            end

          biases_rows * biases_columns == rows
        end,
      {:ok, casted_biases} <- Data.cast(data_type, biases, {rows, 1})
    ) do
      {:ok, :biases, casted_biases}
    end
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

      data_type =
        dense
        |> Dense.data_type()
        |> inspect()

      "#Dense<[#{data_type}, #{shape}]>"
    end
  end
end
