defmodule Annex.Data do
  alias Annex.Encoder

  @behaviour Annex.Encoder

  @type struct_data :: struct()
  @type float_data :: list(float())
  @type data :: struct_data() | float_data()
  @type dataset :: list(data())
  @type t :: dataset() | data()

  @spec encode(Encoder.t()) :: data()
  def encode(%module{} = data), do: module.encode(data)
  def encode(data) when is_list(data), do: data

  @spec decode(Encoder.t()) :: float_data()
  def decode(%module{} = data), do: module.decode(data)
  def decode(data) when is_list(data), do: data
end
