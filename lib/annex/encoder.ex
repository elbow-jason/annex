# defmodule Annex.Encoder do
#   @type t :: module()

#   @callback encode(any()) :: any()
#   @callback decode(any()) :: list(list(float))
#   @callback encoded?(any) :: boolean()

#   @spec encode(module(), any()) :: any()
#   def encode(encoder, data), do: encoder.encode(data)

#   @spec decode(module(), any()) :: list(list(float()))
#   def decode(encoder, data), do: encoder.decode(data)

#   @spec encoded?(module(), any()) :: boolean()
#   def encoded?(encoder, data), do: encoder.encoded?(data)

#   @spec convert(module(), module(), any()) :: any()
#   def convert(source_encoder, destination_encoder, data) do
#     if destination_encoder.encoded?(data) == true do
#       data
#     else
#       data
#       |> source_encoder.decode()
#       |> destination_encoder.encode()
#     end
#   end
# end
