defmodule Annex.Layer.Dropout do
  @moduledoc """
  Given a `frequency` the dropout layer randomly drops an input at the `frequency`.
  """
  alias Annex.{
    Data.List1D,
    Data.Shape,
    Layer,
    Layer.Backprop,
    Layer.Dropout,
    Utils
  }

  @behaviour Layer

  @type t :: %__MODULE__{
          frequency: float()
        }

  defstruct [:frequency]

  @spec build(float()) :: t()
  def build(frequency)
      when is_float(frequency) and frequency >= 0.0 and frequency <= 1.0 do
    %Dropout{frequency: frequency}
  end

  def frequency(%Dropout{frequency: f}), do: f

  @impl Layer
  @spec init_layer(t(), Keyword.t()) :: {:ok, t()}
  def init_layer(%Dropout{} = dropout, _opts \\ []) do
    {:ok, dropout}
  end

  @impl Layer
  @spec feedforward(t(), List1D.t()) :: {t(), List1D.t()}
  def feedforward(%Dropout{} = layer, inputs) do
    {layer, drop(inputs, frequency(layer))}
  end

  @impl Layer
  @spec backprop(t(), List1D.t(), Backprop.t()) :: {t(), List1D.t(), Backprop.t()}
  def backprop(%Dropout{} = dropout, error, backprop), do: {dropout, error, backprop}

  @impl Layer
  @spec data_type :: List1D
  def data_type, do: List1D

  @impl Layer
  @spec shapes(t()) :: {Shape.t(), Shape.t()}
  def shapes(%Dropout{}), do: {:defer, :defer}

  defp drop(inputs, frequency) do
    Enum.map(inputs, fn value -> zeroize_by_frequency(frequency, value) end)
  end

  defp zeroize_by_frequency(frequency, value) do
    if Utils.random_float() <= frequency, do: 0.0, else: value
  end
end
