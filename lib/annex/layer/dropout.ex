defmodule Annex.Layer.Dropout do
  @moduledoc """
  Given a `frequency` the dropout layer randomly drops an input at the `frequency`.
  """
  alias Annex.{
    Layer,
    Layer.Backprop,
    Layer.Dropout,
    Layer.ListLayer,
    Utils
  }

  @behaviour Layer

  use ListLayer

  @type t :: %__MODULE__{
          frequency: float()
        }

  defstruct [:frequency]

  @spec build(float()) :: t()
  def build(frequency)
      when is_float(frequency) and frequency >= 0.0 and frequency <= 1.0 do
    %Dropout{frequency: frequency}
  end

  @spec init_layer(t(), Keyword.t()) :: {:ok, t()}
  def init_layer(%Dropout{} = dropout, _opts \\ []) do
    {:ok, dropout}
  end

  @spec backprop(t(), ListLayer.t(), Backprop.t()) :: {t(), ListLayer.t(), Backprop.t()}
  def backprop(%Dropout{} = dropout, error, backprop) do
    {dropout, error, backprop}
  end

  @spec feedforward(t(), ListLayer.t()) :: {t(), ListLayer.t()}
  def feedforward(%Dropout{frequency: frequency} = layer, inputs) do
    {layer, drop(inputs, frequency)}
  end

  @spec drop(ListLayer.t(), float()) :: ListLayer.t()
  def drop(inputs, frequency) do
    Enum.map(inputs, fn value -> zeroize_by_frequency(frequency, value) end)
  end

  defp zeroize_by_frequency(frequency, value) do
    if Utils.random_float() <= frequency, do: 0.0, else: value
  end
end
