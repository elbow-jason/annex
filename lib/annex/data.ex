defmodule Annex.Data do
  alias Annex.Encoder

  @behaviour Annex.Encoder

  @spec encode(Encoder.t()) :: Encoder.t()
  def encode(%module{} = data), do: module.encode(data)
  def encode(data) when is_list(data), do: data

  @spec decode(Encoder.t()) :: [float(), ...]
  def decode(%module{} = data), do: module.decode(data)
  def decode(data) when is_list(data), do: data
end
