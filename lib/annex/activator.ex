defmodule Annex.Activator do
  alias Annex.Activator

  defstruct [:activation, :derivative, :name]

  @spec build(:relu | :sigmoid | :tanh | {:relu, number()}) :: Annex.Activator.t()
  def build(name) do
    case name do
      {:relu, threshold} ->
        %Activator{
          activation: fn n -> relu_with_threshold(n, threshold) end,
          derivative: fn n -> relu_deriv(n, threshold) end,
          name: name
        }

      :relu ->
        %Activator{
          activation: &relu/1,
          derivative: &relu_deriv/1,
          name: name
        }

      :sigmoid ->
        %Activator{
          activation: &sigmoid/1,
          derivative: &sigmoid_deriv/1,
          name: name
        }

      :tanh ->
        %Activator{
          activation: &tanh/1,
          derivative: &tanh_deriv/1,
          name: name
        }
    end
  end

  def get_activation(%Activator{activation: act}), do: act
  def get_derivative(%Activator{derivative: deriv}), do: deriv

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
