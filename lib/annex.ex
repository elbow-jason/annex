defmodule Annex do
  @moduledoc """
  Annex is a library for composing and running deep artificial
  """

  alias Annex.{
    Data.List1D,
    Layer,
    Layer.Activation,
    Layer.Dense,
    Layer.Dropout,
    Layer.Sequence,
    Learner
  }

  @doc """
  Given a list of `layers` and `opts` returns a built, but not initialized
  `Sequence`.
  """
  @spec sequence(list(struct()), keyword()) :: Sequence.t()
  def sequence(layers, opts \\ []) when is_list(layers) do
    Sequence.build(layers, opts)
  end

  @doc """
  Given a frequency (between `0.0` and `1.0`) returns a `Dropout` layer that
  randomly, at the given frequency, returns `0.0` for an input regardless of
  that input's value.
  """
  @spec dropout(float()) :: Dropout.t()
  def dropout(frequency) do
    Dropout.build(frequency)
  end

  @doc """
  Given a `Learner` or `Layer` behaviour implementing struct initializes the
  struct.

  If the given struct is both a `Learner` and a `Layer` calling `initialize/2`
  will call `Learner.init_learner/2` before `Layer.init_layer/2` .

  If the given struct is not a `Learner` nor a `Layer` this function returns
  an error tuple: `{:error, :invalid_model}`.
  """
  @spec initialize(struct()) :: struct()
  def initialize(model, opts \\ []) do
    [
      {is_learner?(model), &Learner.init_learner/2},
      {is_layer?(model), &Layer.init_layer/2}
    ]
    |> Enum.group_by(fn {k, _} -> k end, fn {_, v} -> v end)
    |> Map.get(true, [])
    |> case do
      [] ->
        {:error, :invalid_model}

      funcs ->
        do_init(model, funcs, opts)
    end
  end

  defp do_init(model, funcs, opts) do
    Enum.reduce_while(funcs, model, fn fun, acc ->
      case fun.(acc, opts) do
        {:ok, acc} -> {:cont, acc}
        {:error, _} = err -> {:halt, err}
      end
    end)
  end

  defp is_layer?(%module{}), do: function_exported?(module, :init_layer, 2)
  defp is_layer?(_), do: false

  defp is_learner?(%module{}), do: function_exported?(module, :init_learner, 2)
  defp is_learner?(_), do: false

  @doc """
  Given a number of `rows`, `columns`, some `weights`,
  and some `biases` returns a built `Dense` layer.
  """
  @spec dense(pos_integer(), pos_integer(), List1D.t(), List1D.t()) :: Dense.t()
  def dense(rows, columns, weights, biases) do
    Dense.build(rows, columns, weights, biases)
  end

  @doc """
  Given a number of `rows` and `columns` returns a Dense layer.

  Without the `weights` and `biases` of `dense/4` this Dense layer will be
  have no neurons. Upon `Layer.init_layer/2` the Dense layer will be
  initialized with random neurons; Neurons with random weights and biases.
  """
  @spec dense(pos_integer(), pos_integer()) :: Dense.t()
  def dense(rows, columns) do
    Dense.build(rows, columns)
  end

  @doc """
  Given a number of `rows` returns a Dense layer.

  Without the `columns` of `dense/2` the `columns` of the returned Dense layer
  must be inferred during initialization (`init_layer/2`) from a previous
  layer's number of outputs.

  Without the `weights` and `biases` of `dense/4` this Dense layer will be
  have no neurons. Upon `Layer.init_layer/2` the Dense layer will be
  initialized with random neurons; Neurons with random weights and biases.
  """
  @spec dense(pos_integer()) :: Dense.t()
  def dense(rows) do
    Dense.build(rows)
  end

  @doc """
  Given an Activation's name returns appropriate `Activation` layer.
  """
  @spec activation(Activation.func_name()) :: Activation.t()
  def activation(name) do
    Activation.build(name)
  end

  @doc """
  Trains an `Annex.Learner` given `learner`, `data`, `labels`, and `options`.

  The `learner` should be initialized `Learner.init_learner/2` before being
  trained.

  Returns the trained `learner` along with some measure of loss or performance.
  """
  @spec train(struct(), Learner.data(), Learner.data(), Keyword.t()) ::
          {:ok, Learner.t(), Learner.data()} | {:error, any()}
  def train(%_{} = learner, data, labels, options \\ []) do
    Learner.train(learner, data, labels, options)
  end

  @doc """
  Given an initialized Learner `learner` and some `data` returns a prediction.

  The `learner` should be initialized with `Learner.init_learner` before being
  used with the `predict/2` function.

  Also, it's a good idea to train the `learner` (using `train/3` or `train/4`)
  before using it to make predicitons. Chances are slim that an untrained
  Learner is capable of making accurate predictions.
  """
  @spec predict(Learner.t(), Learner.data()) :: Learner.data()
  def predict(learner, data) do
    Learner.predict(learner, data)
  end
end
