defmodule Annex.Layer.Backprop do
  alias Annex.{
    Data,
    Defaults,
    Layer.Backprop
  }

  @type derivative :: (float() -> float())
  @type cost_func :: (float() -> float())

  @type t :: %__MODULE__{
          net_loss: float(),
          loss_pds: Data.float_data(),
          derivative: derivative(),
          learning_rate: float(),
          cost_func: cost_func()
        }

  defstruct [:net_loss, :loss_pds, :derivative, :learning_rate, :cost_func]

  def new do
    %Backprop{
      net_loss: 0.0,
      loss_pds: [],
      derivative: Defaults.derivative(),
      cost_func: Defaults.cost(),
      learning_rate: Defaults.learning_rate()
    }
  end

  def get_cost_func(%Backprop{cost_func: cost_func}), do: cost_func

  @spec get_learning_rate(t()) :: float()
  def get_learning_rate(%Backprop{learning_rate: learning_rate}), do: learning_rate

  @spec get_derivative(t()) :: derivative()
  def get_derivative(%Backprop{derivative: derviative}), do: derviative

  @spec get_net_loss(t()) :: float()
  def get_net_loss(%Backprop{net_loss: net_loss}), do: net_loss

  @spec get_loss_pds(t()) :: Data.data()
  def get_loss_pds(%Backprop{loss_pds: loss_pds}), do: loss_pds

  @spec put_learning_rate(t(), float()) :: t()
  def put_learning_rate(%Backprop{} = bp, learning_rate) do
    %Backprop{bp | learning_rate: learning_rate}
  end

  @spec put_derivative(t(), derivative()) :: t()
  def put_derivative(%Backprop{} = bp, derivative) when is_function(derivative, 1) do
    %Backprop{bp | derivative: derivative}
  end

  @spec put_loss_pds(t(), Data.data()) :: t()
  def put_loss_pds(%Backprop{} = bp, loss_pds) do
    %Backprop{bp | loss_pds: loss_pds}
  end

  @spec put_net_loss(t(), any()) :: t()
  def put_net_loss(%Backprop{} = bp, net_loss) do
    %Backprop{bp | net_loss: net_loss}
  end

  def put_cost_func(%Backprop{} = bp, cost) do
    %Backprop{bp | cost_func: cost}
  end
end
