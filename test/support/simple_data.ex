defmodule AnnexHelpers.SimpleData do
  alias AnnexHelpers.SimpleData
  alias Annex.Data

  @behaviour Data

  defstruct [:data, :shape]

  def shape(%SimpleData{shape: shape}), do: shape

  def cast(%SimpleData{} = simple, shape) do
    %SimpleData{simple | shape: shape}
  end

  def cast(data, shape) do
    %SimpleData{
      data: data,
      shape: shape
    }
  end

  def to_flat_list(%SimpleData{data: data}) do
    data
    |> Enum.into([])
    |> List.flatten()
  end
end
