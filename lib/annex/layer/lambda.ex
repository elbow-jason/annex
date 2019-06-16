defmodule Annex.Layer.Lambda do
  @moduledoc """
  `Lambda` is an `Annex.Layer` that contains callbacks that execute at each Layer callback step.
  It can be used to execute arbitrary callback code at each step in this layers Layer
  callbacks allowing execution of arbitrary code.

  It is very useful as a runtime configured layer or a layer that only needs to
  implement

  Technically, any layer could be implemented as a `Lambda` layer.
  """
  alias Annex.Layer.{
    Backprop,
    Lambda
  }

  @type callback2(out) :: (t(), any() -> out)
  @type callback3(out) :: (t(), any(), any() -> out)

  @type t :: %Lambda{
          on_init_layer: callback2({:ok, t()} | {:error, any()}),
          on_feedforward: callback2({t(), any()}),
          on_backprop: callback3({t(), any(), Backprop.t()}),
          on_encoded?: callback2(boolean()),
          on_encode: callback2(any()),
          on_decode: callback2(any()),
          state: any()
        }

  defstruct [
    :on_init_layer,
    :on_feedforward,
    :on_backprop,
    :on_encoded?,
    :on_encode,
    :on_decode,
    :state
  ]

  @behaviour Annex.Layer

  def get_state(%Lambda{state: state}), do: state

  def put_state(%Lambda{} = lambda, state) do
    %Lambda{lambda | state: state}
  end

  def update_state(%Lambda{} = lambda, func) when is_function(func, 1) do
    state =
      lambda
      |> get_state()
      |> func.()

    put_state(lambda, state)
  end

  @spec init_layer(t(), Keyword.t()) :: {:ok, t()} | {:error, any()}
  def init_layer(%Lambda{} = lambda, opts \\ []) do
    apply_callback(lambda, :on_init_layer, [lambda, opts], {:ok, lambda})
  end

  @spec backprop(t(), any, Backprop.t()) :: {t(), any, Backprop.t()}
  def backprop(%Lambda{} = lambda, error, backprop) do
    apply_callback(lambda, :on_backprop, [lambda, error, backprop], {lambda, error, backprop})
  end

  @spec feedforward(t(), any()) :: {t(), any()}
  def feedforward(%Lambda{} = lambda, inputs) do
    apply_callback(lambda, :on_feedforward, [lambda, inputs], {lambda, inputs})
  end

  @spec encode(t(), any) :: any
  def encode(%Lambda{} = lambda, data) do
    apply_callback(lambda, :on_encode, [lambda, data], data)
  end

  @spec encoded?(t(), any) :: boolean()
  def encoded?(%Lambda{} = lambda, data) do
    apply_callback(lambda, :on_encoded?, [lambda, data], true)
  end

  @spec decode(t(), any) :: [float()]
  def decode(%Lambda{} = lambda, data) do
    apply_callback(lambda, :on_decode, [lambda, data], data)
  end

  defp apply_callback(%Lambda{} = lambda, key, args, returning) do
    lambda
    |> Map.fetch!(key)
    |> apply_callback(args, returning)
  end

  defp apply_callback(nil, _args, returning) do
    returning
  end

  defp apply_callback(func, args, _) when is_function(func) do
    apply(func, args)
  end
end
