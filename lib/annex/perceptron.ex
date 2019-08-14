defmodule Annex.Perceptron do
  @moduledoc """
  A simple perceptron Learner capable of making good predictions given a linearly separable
  dataset and labels.
  """
  use Annex.Learner

  alias Annex.{
    Data.List1D,
    Dataset,
    Perceptron,
    Utils
  }

  import Annex.Utils, only: [is_pos_integer: 1]

  @type activation :: (number() -> float())
  @type data_type :: List1D

  @type t :: %Perceptron{
          weights: list(float),
          learning_rate: float(),
          activation: activation(),
          bias: float()
        }
  defstruct [:weights, :learning_rate, :activation, :bias]

  @spec new(pos_integer, activation(), Keyword.t()) :: Perceptron.t()
  def new(inputs, activation, opts \\ [])
      when is_pos_integer(inputs) and is_function(activation, 1) do
    %Perceptron{
      weights: get_or_create_weights(inputs, opts),
      bias: Keyword.get(opts, :bias, 1.0),
      learning_rate: Keyword.get(opts, :learning_rate, 0.05),
      activation: activation
    }
  end

  defp get_or_create_weights(inputs, opts) do
    case Keyword.get(opts, :weights) do
      weights when length(weights) == inputs -> weights
      _ -> List1D.new_random(inputs)
    end
  end

  @spec predict(t(), List1D.t()) :: float()
  def predict(%Perceptron{activation: activation, weights: weights, bias: bias}, inputs) do
    inputs
    |> List1D.dot(weights)
    |> Kernel.+(bias)
    |> activation.()
  end

  @spec train(t(), Dataset.t(), Keyword.t()) :: struct()
  def train(%Perceptron{} = p, dataset, opts \\ []) do
    runs = Keyword.fetch!(opts, :runs)

    fn -> Enum.random(dataset) end
    |> Stream.repeatedly()
    |> Stream.with_index()
    |> Enum.reduce_while(p, fn {data_row, index}, p_acc ->
      if index >= runs do
        {:halt, p_acc}
      else
        {:cont, train_once(p_acc, data_row)}
      end
    end)
  end

  defp train_once(%Perceptron{} = p, {inputs, label}) do
    %Perceptron{
      weights: weights,
      bias: bias,
      learning_rate: lr
    } = p

    prediction = predict(p, inputs)
    error = label - prediction
    slope_delta = error * lr

    updated_weights =
      inputs
      |> Utils.zip(weights)
      |> Enum.map(fn {i, w} -> w + slope_delta * i end)

    %Perceptron{p | weights: updated_weights, bias: bias + slope_delta}
  end
end
