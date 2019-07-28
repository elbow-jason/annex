defmodule AnnexHelpers.SimpleData do
  @moduledoc """
  A simple Annex.Data implementation for use in tests.

  Assumes `data` is a list (because it's simple).
  """
  use Annex.Data

  alias Annex.Data
  alias AnnexHelpers.SimpleData

  @type datum :: float() | [datum]
  @type data :: [datum]

  @type t :: %__MODULE__{
          internal: data,
          shape: Shape.t()
        }

  defstruct [:internal, :shape]

  def get_internal(%SimpleData{internal: internal}) when Data.is_flat_data(internal) do
    internal
  end

  @impl Data
  @spec shape(t()) :: Shape.t()
  def shape(%SimpleData{shape: shape}), do: shape

  @impl Data
  @spec cast(t() | data(), Shape.t()) :: t()
  def cast(%SimpleData{} = simple, shape) do
    simple
    |> get_internal()
    |> cast(shape)
  end

  def cast(data, shape) when Data.is_flat_data(data) do
    product = Shape.product(shape)
    data_size = length(data)

    if product != data_size do
      raise Annex.AnnexError,
        message: """
        SimpleData.cast/2 data size did not match shape size.

        shape_size: #{inspect(product)}
        data_size: #{inspect(data_size)}
        shape: #{inspect(shape)}
        data: #{inspect(data)}
        """
    end

    %SimpleData{
      internal: data,
      shape: shape
    }
  end

  @impl Data
  @spec to_flat_list(t()) :: Data.flat_data()
  def to_flat_list(%SimpleData{internal: internal}) when Data.is_flat_data(internal) do
    internal
  end

  @impl Data
  @spec is_type?(any) :: boolean()
  def is_type?(%SimpleData{}), do: true
  def is_type?(_), do: false

  @impl Data
  @spec apply_op(t(), Data.op(), Data.args()) :: t()
  def apply_op(%SimpleData{} = data, name, args) do
    flat_data = get_internal(data)

    case {name, args} do
      {_, [func]} when is_function(func, 1) ->
        %SimpleData{data | internal: Enum.map(flat_data, func)}
    end
  end
end
