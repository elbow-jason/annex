defmodule Annex.Layer.Dropout do
  @moduledoc """
  Given a `frequency` the dropout layer randomly drops an input at the `frequency`.
  """
  use Annex.Debug, debug: true

  alias Annex.{
    Data,
    Layer,
    Layer.Backprop,
    Layer.Dropout,
    Utils
  }

  use Layer

  @type t :: %__MODULE__{
          frequency: float()
        }

  @type data :: Data.data()

  defstruct [:frequency]

  defguard is_frequency(x) when is_float(x) and x >= 0.0 and x <= 1.0

  @impl Layer
  @spec init_layer(LayerConfig.t(Dropout)) :: t()
  def init_layer(%LayerConfig{} = cfg) do
    cfg
    |> LayerConfig.details()
    |> Map.fetch(:frequency)
    |> case do
      {:ok, frequency} when is_frequency(frequency) ->
        %Dropout{frequency: frequency}

      {:ok, not_frequency} ->
        raise %AnnexError{
          message: "Dropout.build/1 requires a :frequency that is a float between 0.0 and 1.0",
          details: [
            invalid_frequency: not_frequency,
            reason: :invalid_frequency_value
          ]
        }

      :error ->
        raise %AnnexError{
          message: "Dropout.build/1 requires a :frequency that is a float between 0.0 and 1.0",
          details: [
            reason: {:key_not_found, :frequency}
          ]
        }
    end
  end

  @spec frequency(t()) :: float()
  def frequency(%Dropout{frequency: f}), do: f

  # @impl Layer
  # @spec init_layer(t(), Keyword.t()) :: {:ok, t()}
  # def init_layer(%Dropout{} = dropout, _opts \\ []) do
  #   {:ok, dropout}
  # end

  @impl Layer
  @spec feedforward(t(), data()) :: {t(), data()}
  def feedforward(%Dropout{} = layer, inputs) do
    {layer, drop(inputs, frequency(layer))}
  end

  @impl Layer
  @spec backprop(t(), data(), Backprop.t()) :: {t(), data(), Backprop.t()}
  def backprop(%Dropout{} = dropout, error, backprop), do: {dropout, error, backprop}

  defp drop(inputs, frequency) do
    data_type = Data.infer_type(inputs)
    dropper = fn value -> zeroize_by_frequency(frequency, value) end
    Data.apply_op(data_type, inputs, :map, [dropper])
  end

  defp zeroize_by_frequency(frequency, value) do
    if Utils.random_float() <= frequency, do: 0.0, else: value
  end
end
