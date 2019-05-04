defmodule Annex.Layer.Dropout do
  alias Annex.{Layer, Layer.Backprop, Layer.Dropout, Utils}

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

  @spec init_layer(t(), Keyword.t()) :: {:ok, t()}
  def init_layer(%Dropout{} = dropout, _opts \\ []) do
    {:ok, dropout}
  end

  @spec backprop(t(), Backprop.t()) :: {t(), Backprop.t()}
  def backprop(%Dropout{} = dropout, backprop) do
    {dropout, backprop}
  end

  @spec feedforward(t(), list(float())) :: {list(float()), t()}
  def feedforward(%Dropout{frequency: frequency} = dropout, inputs) do
    {Enum.map(inputs, fn value -> zeroize_by_frequency(frequency, value) end), dropout}
  end

  @spec encoder() :: Annex.Data
  def encoder, do: Annex.Data

  defp zeroize_by_frequency(frequency, value) do
    if Utils.random_float() <= frequency, do: 0.0, else: value
  end
end
