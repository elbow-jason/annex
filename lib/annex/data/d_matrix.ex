defmodule Annex.Data.DMatrix do
  @moduledoc """
  A 2D Matrix Annex.Data that uses the Tensor library as the underlying data structure.

  For the purpose of Dense Matrices it seems to be slower than a flat list or a list of lists.
  """
  use Annex.Debug, debug: true
  use Tensor
  use Annex.Data

  alias Annex.{
    Data,
    Data.DMatrix,
    Utils
  }

  @type tensor :: any()
  @type t :: %__MODULE__{tensor: tensor()}
  @type rows :: pos_integer()
  @type columns :: pos_integer()
  @type op :: :dot | :add | :subtract | :multiply

  defstruct [:tensor]

  @impl Data
  @spec cast(Data.flat_data() | t(), {pos_integer, pos_integer}) :: t()
  def cast(data, shape) when Data.is_flat_data(data) do
    case shape do
      {rows, columns} ->
        build(data, rows, columns)

      {columns} ->
        build(data, 1, columns)
    end
  end

  def cast(%DMatrix{} = dmatrix, {rows, columns}) do
    case shape(dmatrix) do
      {^rows, ^columns} ->
        dmatrix

      _ ->
        dmatrix
        |> to_flat_list()
        |> cast({rows, columns})
    end
  end

  def cast(%DMatrix{} = dmatrix, {columns}) do
    dmatrix
    |> to_flat_list()
    |> cast({1, columns})
  end

  @impl Data
  @spec is_type?(any) :: boolean
  def is_type?(%DMatrix{}), do: true
  def is_type?(_), do: false

  @impl Data
  @spec to_flat_list(t()) :: [float()]
  def to_flat_list(%DMatrix{} = dmatrix) do
    dmatrix
    |> to_list_of_lists()
    |> List.flatten()
  end

  @impl Data
  @spec shape(t()) :: {pos_integer(), pos_integer()}
  def shape(%DMatrix{} = dmatrix) do
    dmatrix
    |> tensor()
    |> Map.fetch!(:dimensions)
    |> List.to_tuple()
  end

  @impl Data
  @spec apply_op(t(), Data.op(), Data.args()) :: t()
  def apply_op(data, op, extra_args) when is_list(extra_args) do
    case {op, extra_args} do
      {:dot, [right]} -> dot(data, right)
      {:add, [right]} -> add(data, right)
      {:subtract, [right]} -> subtract(data, right)
      {:multiply, [right]} -> multiply(data, right)
      {:transpose, []} -> transpose(data)
      {_, [func]} when is_function(func, 1) -> map(data, func)
    end
  end

  @spec new_random(rows(), columns()) :: t()
  def new_random(rows, columns) do
    product = rows * columns

    product
    |> Utils.random_weights()
    |> build_tensor(rows, columns)
    |> from_tensor()
  end

  @spec build([float(), ...] | [[float, ...], ...]) :: Annex.Data.DMatrix.t()
  def build([f | _] = data) when is_float(f) do
    columns = length(data)
    build(data, 1, columns)
  end

  def build([[f | _] = first_row | _] = data) when is_float(f) do
    rows = length(data)
    columns = length(first_row)

    data
    |> List.flatten()
    |> build(rows, columns)
  end

  @spec build([float(), ...], non_neg_integer, pos_integer) :: t()
  def build([f | _] = data, rows, columns) when is_float(f) do
    size = length(data)
    product = rows * columns

    if size != product do
      raise ArgumentError,
        message: """
        DMatrix.build/3 expects the product of rows * columns to equal the length of the data.

        data_length: #{inspect(size)}
        product: #{inspect(product)}

        rows: #{inspect(rows)}
        columns: #{inspect(columns)}
        data: #{inspect(data)}
        """
    end

    data
    |> build_tensor(rows, columns)
    |> from_tensor()
  end

  defp build_tensor(data, rows, columns) do
    data
    |> Enum.chunk_every(columns)
    |> Matrix.new(rows, columns)
  end

  @spec ones(non_neg_integer, non_neg_integer) :: t()
  def ones(rows, columns) do
    %DMatrix{tensor: Matrix.new([], rows, columns, 1.0)}
  end

  @spec zeros(non_neg_integer, non_neg_integer) :: t()
  def zeros(rows, columns) do
    %DMatrix{tensor: Matrix.new([], rows, columns, 0.0)}
  end

  @spec tensor(t()) :: any()
  def tensor(%DMatrix{tensor: tensor}), do: tensor

  @spec to_list_of_lists(t()) :: [[float(), ...], ...]
  def to_list_of_lists(%DMatrix{} = dmatrix) do
    dmatrix
    |> tensor()
    |> Tensor.to_list()
  end

  @spec dot(t(), t()) :: t()
  def dot(%DMatrix{} = left, %DMatrix{} = right) do
    debug_assert "left DMatrix must be 2D" do
      left_shape = shape(left)
      tuple_size(left_shape) == 2
    end

    debug_assert "right DMatrix must be 2D" do
      right_shape = shape(right)
      tuple_size(right_shape) == 2
    end

    debug_assert "DMatrix.dot/2 - left columns must the the same as right rows" do
      {_, left_columns} = shape(left)
      {right_rows, _} = shape(right)

      left_columns == right_rows
    end

    apply_tensor(left, right, &Matrix.product/2)
  end

  def dot(%DMatrix{} = left, data) when Data.is_flat_data(data) do
    {_, columns} = shape(left)
    # build it so that dot can be performed.
    # in the future we might need to cast the shape with the
    # given rows as well...
    right = DMatrix.build(data, columns, 1)
    apply_tensor(left, right, &Matrix.product/2)
  end

  @spec multiply(t(), t() | number | [float(), ...]) :: t()
  def multiply(%DMatrix{} = d, n) when is_number(n) do
    apply_tensor(d, n, &Matrix.mult_number/2)
  end

  def multiply(%DMatrix{} = left, data) when Data.is_flat_data(data) do
    {rows, columns} = shape(left)
    # build it the same shape for multiply
    right = DMatrix.build(data, rows, columns)

    multiply(left, right)
  end

  def multiply(%DMatrix{} = left, %DMatrix{} = right) do
    debug_assert "shapes must match for mutltiply" do
      left_shape = shape(left)
      right_shape = shape(right)
      left_shape == right_shape
    end

    apply_tensor(left, right, &Matrix.mult_matrix/2)
  end

  @spec add(t(), number | t()) :: t()
  def add(%DMatrix{} = d, n) when is_number(n) do
    apply_tensor(d, n, &Matrix.add_number/2)
  end

  def add(%DMatrix{} = left, %DMatrix{} = right) do
    debug_assert "shapes must match for add" do
      left_shape = shape(left)
      right_shape = shape(right)
      left_shape == right_shape
    end

    apply_tensor(left, right, &Matrix.add_matrix/2)
  end

  @spec subtract(t(), number | t()) :: t()
  def subtract(%DMatrix{} = left, right) when is_number(right) do
    apply_tensor(left, right, &Matrix.sub_number/2)
  end

  def subtract(%DMatrix{} = left, %DMatrix{} = right) do
    debug_assert "shapes must match for subtract" do
      left_shape = shape(left)
      right_shape = shape(right)
      left_shape == right_shape
    end

    apply_tensor(left, right, &Matrix.sub_matrix/2)
  end

  @spec transpose(t()) :: t()
  def transpose(%DMatrix{} = d) do
    apply_tensor(d, &Matrix.transpose/1)
  end

  def map(%DMatrix{} = d, fun) do
    apply_tensor(d, fn tensor -> Matrix.map(tensor, fun) end)
  end

  defp from_tensor(tensor) do
    %DMatrix{tensor: tensor}
  end

  defp apply_tensor(%DMatrix{} = left, %DMatrix{} = right, fun) do
    left
    |> tensor()
    |> fun.(tensor(right))
    |> from_tensor()
  end

  defp apply_tensor(%DMatrix{} = left, right, fun) do
    left
    |> tensor()
    |> fun.(right)
    |> from_tensor()
  end

  defp apply_tensor(%DMatrix{} = d, fun) do
    d
    |> tensor()
    |> fun.()
    |> from_tensor()
  end

  defimpl Inspect do
    def inspect(dmatrix, _) do
      "#DMatrix<#{do_shape(dmatrix)}, #{do_values(dmatrix)}>"
    end

    defp do_shape(dmatrix) do
      dmatrix
      |> DMatrix.shape()
      |> inspect()
    end

    def do_values(dmatrix) do
      dmatrix
      |> DMatrix.to_list_of_lists()
      |> inspect()
    end
  end

  defimpl Enumerable do
    alias Annex.Data.DMatrix

    @spec count(DMatrix.t()) :: {:ok, pos_integer}
    def count(dmatrix) do
      product =
        dmatrix
        |> DMatrix.tensor()
        |> DMatrix.shape()
        |> Shape.product()

      {:ok, product}
    end

    def member?(_dmatrix, _element), do: {:error, __MODULE__}

    def reduce(dmatrix, acc, fun) do
      dmatrix
      |> DMatrix.tensor()
      |> Matrix.to_list()
      |> List.flatten()
      |> do_reduce(acc, fun)
    end

    defdelegate do_reduce(list, signal, fun), to: Enumerable.List, as: :reduce

    def slice(_tensor) do
      {:error, __MODULE__}
    end
  end
end
