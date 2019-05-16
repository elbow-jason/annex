# defmodule Annex.ListOfFloats do
#   @behaviour Annex.Encoder

#   @type t :: list(list(float))

#   @spec encoded?(any()) :: boolean()
#   def encoded?([f | _]) when is_float(f), do: true
#   def encoded?(_), do: false

#   @spec encode(any()) :: t()
#   def encode([f | _] = data) when is_float(f), do: data
#   def encode(data) when is_list(data), do: data |> List.flatten() |> encode()
#   def encode(data), do: data |> Enum.to_list() |> encode()

#   @spec decode(t() | list(float())) :: t()
#   def decode([[f | _] | _] = lol) when is_float(f), do: lol
#   def decode(data), do: data |> encode() |> decode()
# end
