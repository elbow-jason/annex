defmodule Annex.Activation do
  alias Annex.{Activation, Layer, Backprop}

  @type t :: %__MODULE__{
          activator: (number -> number),
          derivative: (number -> number),
          name: atom()
        }

  @behaviour Layer

  defstruct [:activator, :derivative, :name, :inputs, :output]

  @spec build(:relu | :sigmoid | :tanh | {:relu, number()}) :: Annex.Activation.t()
  def build(name) do
    case name do
      {:relu, threshold} ->
        %Activation{
          activator: fn n -> relu_with_threshold(n, threshold) end,
          derivative: fn n -> relu_deriv(n, threshold) end,
          name: name
        }

      :relu ->
        %Activation{
          activator: &relu/1,
          derivative: &relu_deriv/1,
          name: name
        }

      :sigmoid ->
        %Activation{
          activator: &sigmoid/1,
          derivative: &sigmoid_deriv/1,
          name: name
        }

      :tanh ->
        %Activation{
          activator: &tanh/1,
          derivative: &tanh_deriv/1,
          name: name
        }
    end
  end

  def feedforward(%Activation{} = layer, inputs) do
    output = generate_outputs(layer, inputs)
    {output, %Activation{layer | inputs: inputs, output: output}}
  end

  @spec backprop(t(), Backprop.t()) :: {t(), Backprop.t()}
  def backprop(%Activation{} = layer, backprops) do
    {layer, put_backprop_derivative(layer, backprops)}
  end

  defp put_backprop_derivative(layer, backprops) do
    Backprop.put_derivative(backprops, get_derivative(layer))
  end

  def encoder, do: Annex.Data

  def init_layer(%Activation{} = layer, _opts) do
    {:ok, layer}
  end

  def generate_outputs(%Activation{} = act, inputs) do
    Enum.map(inputs, get_activator(act))
  end

  @spec get_activator(Activation.t()) :: (number() -> number())
  def get_activator(%Activation{activator: act}), do: act

  @spec get_derivative(Activation.t()) :: (number() -> number())
  def get_derivative(%Activation{derivative: deriv}), do: deriv

  def relu(n) do
    relu_with_threshold(n, 0.0)
  end

  def relu_deriv(x), do: relu_deriv(x, 0.0)
  def relu_deriv(x, threshold) when x > threshold, do: 1.0
  def relu_deriv(_, _), do: 0.0

  def relu_with_threshold(n, threshold) do
    if n > threshold do
      n
    else
      threshold
    end
  end

  def sigmoid(n) do
    1.0 / (1.0 + :math.exp(-n))
  end

  def sigmoid_deriv(x) do
    fx = sigmoid(x)
    fx * (1 - fx)
  end

  def tanh(n) do
    :math.tanh(n)
  end

  def tanh_deriv(x) do
    tanh_squared =
      x
      |> :math.tanh()
      |> :math.pow(2)

    1.0 - tanh_squared
  end
end
