# defmodule Annex.Layer.Neuron do
#   @moduledoc """
#   The Neuron itself is not a Layer, but a helper model for the Dense Layer.

#   It represents the unit of computation for an input.
#   """

#   alias Annex.{Layer.Neuron, Utils}

#   @type t :: %__MODULE__{
#           weights: list(float),
#           bias: float()
#         }

#   defstruct weights: [],
#             bias: 1.0

#   def new(weights, bias) do
#     %Neuron{
#       weights: weights,
#       bias: bias
#     }
#   end

#   def new_random(size) when is_integer(size) and size > 0 do
#     weights = Utils.random_weights(size)
#     bias = Utils.random_float()
#     new(weights, bias)
#   end

#   def get_bias(%Neuron{bias: bias}), do: bias
#   def get_weights(%Neuron{weights: w}), do: w

#   def feedforward(%Neuron{} = neuron, inputs) do
#     neuron
#     |> get_weights
#     |> Enum.zip(inputs)
#     |> Enum.map(fn {w, i} -> w * i end)
#     |> Enum.sum()
#     |> Kernel.+(get_bias(neuron))
#   end

#   @spec backprop(t(), [float()], float(), float(), float(), number) :: {[float()], t()}
#   def backprop(
#         %Neuron{} = neuron,
#         input,
#         sum_deriv,
#         negative_gradient,
#         neuron_error,
#         learning_rate
#       ) do
#     weights = get_weights(neuron)
#     bias = get_bias(neuron)

#     delta_coeff = learning_rate * negative_gradient * neuron_error

#     {[_ | next_neuron_loss], [new_bias | new_weights]} =
#       [1.0 | input]
#       |> Utils.zip([bias | weights])
#       |> Enum.map(fn {input, weight} ->
#         delta = input * sum_deriv * delta_coeff
#         {weight * sum_deriv, weight - delta}
#       end)
#       |> Enum.unzip()

#     {next_neuron_loss, %Neuron{neuron | weights: new_weights, bias: new_bias}}
#   end
# end
