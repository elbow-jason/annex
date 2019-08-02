defmodule Annex.Layer.Activation do
  @moduledoc """
  The Activation layer is the Annex.Layer that is responsible for
  applying an activation function to the data during the feedforward
  and supplying the gradient function (derivative) of the activation
  function to the Backprops during backpropagation.
  """
  use Annex.Debug, debug: true

  alias Annex.Data
  alias Annex.Layer
  alias Annex.LayerConfig
  alias Annex.Layer.Activation
  alias Annex.Layer.Backprop

  @type func_type :: :float | :list

  @type func_name :: :relu | :sigmoid | :tanh | {:relu, number()}

  @type t :: %__MODULE__{
          activator: (number -> number),
          derivative: (number -> number),
          func_type: func_type(),
          name: atom(),
          initialized?: boolean()
        }

  @type data_type :: Data.type()
  @type data :: Data.data()

  @behaviour Layer

  defstruct [
    :activator,
    :derivative,
    :name,
    :outputs,
    :inputs,
    :func_type,
    initialized?: false
  ]

  @impl Layer
  @spec build(LayerConfig.deatils()) :: t()
  def build(details) when is_map(details) do
    case details do
      %{name: name} -> from_name(name)
    end
  end

  @spec from_name(:relu | :sigmoid | :softmax | :tanh | {:relu, any}) :: t()
  def from_name(name) do
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
  def init_layer(%Activation{} = activation1, _opts) do
    activation2 = %Activation{activation1 | initialized?: true}

    {:ok, activation2}
  end

  @impl Layer
  @spec feedforward(t(), data()) :: {t(), data()}
  def feedforward(%Activation{} = layer, inputs) do
    outputs = generate_outputs(layer, inputs)
    {%Activation{layer | outputs: outputs, inputs: inputs}, outputs}
  end

  @impl Layer
  @spec backprop(t(), Data.data(), Backprop.t()) :: {t(), Data.data(), Backprop.t()}
  def backprop(%Activation{} = layer, error, props) do
    derivative = get_derivative(layer)

    # name = get_name(layer)
    # next_error =
    #   layer
    #   |> get_inputs()
    #   |> Data.apply_op({:derivative, name}, [derivative])
    #   |> Data.apply_op(:multiply, [error])

    {layer, error, Backprop.put_derivative(props, derivative)}
  end

  @impl Layer
  @spec data_type(t()) :: nil
  def data_type(%Activation{}), do: nil

  @doc """
  Activation has no shape.
  """
  @impl Layer
  @spec shape(t()) :: nil
  def shape(%Activation{}), do: nil

  @spec generate_outputs(t(), Data.data()) :: Data.data()
  def generate_outputs(%Activation{} = layer, inputs) do
    activation = get_activator(layer)
    name = get_name(layer)
    Data.apply_op(inputs, name, [activation])
  end

  @spec get_activator(t()) :: (number() -> number())
  def get_activator(%Activation{activator: act}), do: act

  @spec get_derivative(t()) :: any()
  def get_derivative(%Activation{derivative: deriv}), do: deriv

  @spec get_name(t()) :: any()
  def get_name(%Activation{name: name}), do: name

  @spec get_inputs(t()) :: Data.data()
  def get_inputs(%Activation{inputs: inputs}), do: inputs

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

  @spec softmax(data()) :: data()
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

  defimpl Inspect do
    @spec inspect(Activation.t(), any) :: String.t()
    def inspect(%Activation{name: name}, _) do
      "#Activation<[#{Kernel.inspect(name)}]>"
    end
  end
end
