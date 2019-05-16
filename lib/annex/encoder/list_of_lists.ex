# defmodule Annex.ListOfLists do
#   @behaviour Annex.Encoder

#   @type t :: list(list(float))

#   @spec encoded?(any()) :: boolean()
#   def encoded?([[f | _] | _]) when is_float(f), do: true
#   def encoded?(_), do: false

#   @spec encode(t() | list(float)) :: t()
#   def encode([f | _] = data) when is_float(f), do: [data]
#   def encode([[f | _] | _] = lol) when is_float(f), do: lol

#   @spec decode(t() | list(float())) :: t()
#   def decode([[f | _] | _] = lol) when is_float(f), do: lol
#   def decode(data), do: data |> encode() |> decode()
# end
