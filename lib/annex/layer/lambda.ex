defmodule Annex.Layer.Lambda do
  @moduledoc """
  `Lambda` is an `Annex.Layer` that contains callbacks that execute at each Layer callback step.
  It can be used to execute arbitrary callback code at each step in this layers Layer
  callbacks allowing execution of arbitrary code.

  It is very useful as a runtime configured layer or a layer that only needs to
  implement

  Technically, any layer could be implemented as a `Lambda` layer.
  """
  alias Annex.{
    Data,
    Data.Shape,
    Layer,
    Layer.Backprop,
    Layer.Lambda
  }

  @type callback2(out) :: (t(), any() -> out)
  @type callback3(out) :: (t(), any(), any() -> out)

  @type t :: %Lambda{
          on_init_layer: callback2({:ok, t()} | {:error, any()}) | nil,
          on_feedforward: callback2({t(), any()}) | nil,
          on_backprop: callback3({t(), any(), Backprop.t()}) | nil,
          on_shapes: (t() -> {Shape.t(), Shape.t()}) | nil,
          data_type: Data.type() | nil,
          in_shape: Shape.t(),
          out_shape: Shape.t(),
          state: any()
        }

  defstruct in_shape: :defer,
            out_shape: :defer,
            data_type: nil,
            on_init_layer: nil,
            on_feedforward: nil,
            on_backprop: nil,
            on_shapes: nil,
            state: nil

  def get_state(%Lambda{state: state}), do: state

  def put_state(%Lambda{} = lambda, state), do: %Lambda{lambda | state: state}

  def in_shape(%Lambda{in_shape: s}), do: s

  def out_shape(%Lambda{out_shape: s}), do: s

  def update_state(%Lambda{} = lambda, func) when is_function(func, 1) do
    state =
      lambda
      |> get_state()
      |> func.()

    put_state(lambda, state)
  end

  @behaviour Layer

  @impl Layer
  @spec init_layer(t(), Keyword.t()) :: {:ok, t()} | {:error, any()}
  def init_layer(%Lambda{} = lambda, opts \\ []) do
    apply_callback(lambda, :on_init_layer, [lambda, opts], {:ok, lambda})
  end

  @impl Layer
  @spec feedforward(t(), any()) :: {t(), any()}
  def feedforward(%Lambda{} = lambda, inputs) do
    apply_callback(lambda, :on_feedforward, [lambda, inputs], {lambda, inputs})
  end

  @impl Layer
  @spec backprop(t(), any, Backprop.t()) :: {t(), any, Backprop.t()}
  def backprop(%Lambda{} = lambda, error, backprop) do
    apply_callback(lambda, :on_backprop, [lambda, error, backprop], {lambda, error, backprop})
  end

  @impl Layer
  @spec data_type :: :defer
  def data_type, do: :defer

  @impl Layer
  @spec shapes(t()) :: {Shape.t(), Shape.t()}
  def shapes(%Lambda{} = lambda) do
    apply_callback(lambda, :on_shapes, [lambda], {in_shape(lambda), out_shape(lambda)})
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
