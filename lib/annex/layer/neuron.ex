defmodule Annex.Layer.Neuron do
  alias Annex.{Layer.Neuron, Utils}

  @type t :: %__MODULE__{
          weights: list(float),
          bias: float()
          # sum: float(),
          # output: float(),
          # inputs: list(float)
        }

  defstruct weights: [],
            bias: 1.0

  # sum: 0.0,
  # output: 0.0,
  # inputs: nil

  def new(weights, bias) do
    %Neuron{
      weights: weights,
      bias: bias
    }
  end

  def new_random(size) when is_integer(size) and size > 0 do
    weights = Utils.random_weights(size)
    bias = Utils.random_float()
    new(weights, bias)
  end

  def get_bias(%Neuron{bias: bias}), do: bias
  def get_weights(%Neuron{weights: w}), do: w
  # def get_output(%Neuron{output: o}), do: o
  # def get_sum(%Neuron{sum: sum}), do: sum
  # def get_inputs(%Neuron{inputs: inputs}), do: inputs

  def feedforward(%Neuron{} = neuron, inputs) do
    neuron
    |> get_weights
    |> Enum.zip(inputs)
    |> Enum.map(fn {w, i} -> w * i end)
    |> Enum.sum()
    |> Kernel.+(get_bias(neuron))
  end

  # @spec backprop(t(), float(), float(), float(), (float() -> float())) ::
  #         {list(float()), t()}
  @spec backprop(Annex.Layer.Neuron.t(), [float()], float(), float(), float(), number) ::
          {[float()], Annex.Layer.Neuron.t()}
  def backprop(%Neuron{} = neuron, input, sum_deriv, total_loss_pd, neuron_loss_pd, learning_rate) do
    weights = get_weights(neuron)
    bias = get_bias(neuron)

    # sum = get_sum(neuron)

    # sum_deriv = activation_deriv.(sum)
    # inputs = get_inputs(neuron)

    delta_coeff = learning_rate * total_loss_pd * neuron_loss_pd

    {[_ | next_neuron_loss], [new_bias | new_weights]} =
      [1.0 | input]
      |> Utils.zip([bias | weights])
      |> Enum.map(fn {input, weight} ->
        delta = input * sum_deriv * delta_coeff
        {weight * sum_deriv, weight - delta}
      end)
      |> Enum.unzip()

    {next_neuron_loss, %Neuron{neuron | weights: new_weights, bias: new_bias}}
  end
end
