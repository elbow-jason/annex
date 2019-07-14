defmodule Annex.Layer.Activation do
  @moduledoc """
  The Activation layer is the Annex.Layer that is responsible for
  applying an activation function to the data during the feedforward
  and supplying the gradient function (derivative) of the activation
  function to the Backprops during backpropagation.
  """

  alias Annex.{
    Data.List1D,
    Data.Shape,
    Layer,
    Layer.Activation,
    Layer.Backprop
  }

  @type func_type :: :float | :list

  @type func_name :: :relu | :sigmoid | :tanh | {:relu, number()}

  @type t :: %__MODULE__{
          activator: (number -> number),
          derivative: (number -> number),
          func_type: func_type(),
          name: atom()
        }

  @behaviour Layer

  defstruct [:activator, :derivative, :name, :output, :func_type]

  @spec build(:relu | :sigmoid | :tanh | {:relu, number()}) :: t()
  def build(name) do
    case name do
      {:relu, threshold} ->
        %Activation{
          activator: fn n -> max(n, threshold) end,
          derivative: fn n -> relu_deriv(n, threshold) end,
          func_type: :float,
          name: name
        }

      :relu ->
        %Activation{
          activator: &relu/1,
          derivative: &relu_deriv/1,
          func_type: :float,
          name: name
        }

      :sigmoid ->
        %Activation{
          activator: &sigmoid/1,
          derivative: &sigmoid_deriv/1,
          func_type: :float,
          name: name
        }

      :tanh ->
        %Activation{
          activator: &tanh/1,
          derivative: &tanh_deriv/1,
          func_type: :float,
          name: name
        }

      :softmax ->
        %Activation{
          activator: &softmax/1,
          derivative: &tanh_deriv/1,
          func_type: :list,
          name: name
        }
    end
  end

  @impl Layer
  @spec init_layer(t(), Keyword.t()) :: {:ok, t()}
  def init_layer(%Activation{} = layer, _opts) do
    {:ok, layer}
  end

  @impl Layer
  @spec feedforward(t(), List1D.t()) :: {t(), List1D.t()}
  def feedforward(%Activation{} = layer, inputs) do
    output = generate_outputs(layer, inputs)
    {%Activation{layer | output: output}, output}
  end

  @impl Layer
  @spec backprop(t(), List1D.t(), Backprop.t()) :: {t(), List1D.t(), Backprop.t()}
  def backprop(%Activation{} = layer, error, props) do
    derviative = get_derivative(layer)
    {layer, error, Backprop.put_derivative(props, derviative)}
  end

  @impl Layer
  @spec data_type :: List1D
  def data_type, do: List1D

  @impl Layer
  @spec shapes(t()) :: {Shape.t(), Shape.t()}
  def shapes(%Activation{}), do: {:defer, :defer}

  @spec generate_outputs(t(), List1D.t()) :: [any()]
  def generate_outputs(%Activation{} = layer, inputs) do
    activation = get_activator(layer)
    Enum.map(inputs, activation)
  end

  @spec get_activator(Activation.t()) :: (number() -> number())
  def get_activator(%Activation{activator: act}), do: act

  @spec get_derivative(Activation.t()) :: (number() -> number())
  def get_derivative(%Activation{derivative: deriv}), do: deriv

  @spec relu(float()) :: float()
  def relu(n), do: max(n, 0.0)

  @spec relu_deriv(float()) :: float()
  def relu_deriv(x), do: relu_deriv(x, 0.0)

  @spec relu_deriv(float(), float()) :: float()
  def relu_deriv(x, threshold) when x > threshold, do: 1.0
  def relu_deriv(_, _), do: 0.0

  @spec sigmoid(float()) :: float()
  def sigmoid(n) when n > 100, do: 1.0
  def sigmoid(n) when n < -100, do: 0.0

  def sigmoid(n) do
    1.0 / (1.0 + :math.exp(-n))
  end

  @spec sigmoid_deriv(float()) :: float()
  def sigmoid_deriv(x) do
    fx = sigmoid(x)
    fx * (1.0 - fx)
  end

  @spec softmax(List1D.t()) :: List1D.t()
  def softmax(values) when is_list(values) do
    exps = Enum.map(values, fn vx -> :math.exp(vx) end)
    exps_sum = Enum.sum(exps)
    Enum.map(exps, fn e -> e / exps_sum end)
  end

  @spec tanh(float()) :: float()
  def tanh(n) do
    :math.tanh(n)
  end

  @spec tanh_deriv(float()) :: float()
  def tanh_deriv(x) do
    1.0 - (x |> :math.tanh() |> :math.pow(2))
  end
end
