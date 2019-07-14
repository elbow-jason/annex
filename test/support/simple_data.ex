defmodule AnnexHelpers.SimpleData do
  @moduledoc """
  A simple Annex.Data implementation for use in tests.

  Assumes `data` is a list (because it's simple).
  """
  use Annex.Data

  alias AnnexHelpers.SimpleData

  @type datum :: float() | [datum]
  @type data :: [datum]

  @type t :: %__MODULE__{
          data: data,
          shape: Data.shape()
        }

  defstruct [:data, :shape]

  @spec shape(t()) :: Shape.t()
  def shape(%SimpleData{shape: shape}), do: shape

  @spec cast(t() | data(), Data.shape()) :: t()
  def cast(%SimpleData{} = simple, shape) do
    %SimpleData{simple | shape: shape}
  end

  def cast(data, shape) when is_list(data) do
    %SimpleData{
      data: data,
      shape: shape
    }
  end

  @spec to_flat_list(t()) :: Data.flat_data()
  def to_flat_list(%SimpleData{data: data}) do
    data
    |> Enum.into([])
    |> List.flatten()
  end

  @spec is_type?(any) :: boolean()
  def is_type?(%SimpleData{}), do: true
  def is_type?(_), do: false
end
