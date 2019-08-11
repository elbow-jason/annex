defmodule Annex do
  @moduledoc """
  Annex is a library for composing and running deep artificial
  """

  alias Annex.{
    Data,
    Layer.Activation,
    Layer.Dense,
    Layer.Dropout,
    Layer.Sequence,
    LayerConfig,
    Learner
  }

  @doc """
  Given a list of `layers` returns a `LayerConfig` for a `Sequence`.
  """
  @spec sequence(list(LayerConfig.t(module()))) :: LayerConfig.t(Sequence)
  def sequence(layers) when is_list(layers) do
    LayerConfig.build(Sequence, layers: layers)
  end

  @doc """
  Given a frequency (between `0.0` and `1.0`) returns a `Dropout` layer that
  randomly, at the given frequency, returns `0.0` for an input regardless of
  that input's value.
  """
  @spec dropout(float()) :: LayerConfig.t(Dropout)
  def dropout(frequency) do
    LayerConfig.build(Dropout, frequency: frequency)
  end

  # @doc """
  # Given a `Learner` or `Layer` behaviour implementing struct initializes the
  # struct.

  # If the given struct is both a `Learner` and a `Layer` calling `initialize/2`
  # will call `Learner.init_learner/2` before `Layer.init_layer/2` .

  # If the given struct is not a `Learner` nor a `Layer` this function returns
  # an error tuple: `{:error, :invalid_model}`.
  # """
  # @spec initialize(struct()) :: struct()
  # def initialize(model, opts \\ []) do
  #   cond do
  #     is_learner?(model) -> Learner.init_learner(model, opts)
  #     is_layer_config?(model) -> LayerConfig.init_layer(model)
  #     is_layer?(model) -> model
  #   end
  # end

  # defp is_layer_config?(cfg), do: match?(%LayerConfig{}, cfg)
  # defp is_layer?(layer), do: Layer.is_layer?(layer)
  # defp is_learner?(learner), do: Learner.is_learner?(learner)

  @doc """
  Given a number of `rows`, `columns`, some `weights`,
  and some `biases` returns a built `Dense` layer.
  """
  @spec dense(pos_integer(), pos_integer(), Data.data(), Data.data()) :: LayerConfig.t(Dense)
  def dense(rows, columns, weights, biases) do
    LayerConfig.build(Dense, rows: rows, columns: columns, weights: weights, biases: biases)
  end

  @doc """
  Given a number of `rows` and `columns` returns a Dense layer.

  Without the `weights` and `biases` of `dense/4` this Dense layer will be
  have no neurons. Upon `Layer.init_layer/2` the Dense layer will be
  initialized with random neurons; Neurons with random weights and biases.
  """
  @spec dense(pos_integer(), pos_integer()) :: LayerConfig.t(Dense)
  def dense(rows, columns) do
    LayerConfig.build(Dense, rows: rows, columns: columns)
  end

  # @doc """
  # Given a number of `rows` returns a Dense layer.

  # Without the `columns` of `dense/2` the `columns` of the returned Dense layer
  # must be inferred during initialization (`init_layer/2`) from a previous
  # layer's number of outputs.

  # Without the `weights` and `biases` of `dense/4` this Dense layer will be
  # have no neurons. Upon `Layer.init_layer/2` the Dense layer will be
  # initialized with random neurons; Neurons with random weights and biases.
  # """
  # @spec dense(pos_integer()) :: LayerConfig.t(Dense)
  # def dense(rows) do
  #   LayerConfig.build(Dense, rows: rows)
  # end

  @doc """
  Given an Activation's name returns appropriate `Activation` layer.
  """
  @spec activation(Activation.func_name()) :: LayerConfig.t(Activation)
  def activation(name) do
    LayerConfig.build(Activation, %{name: name})
  end

  @doc """
  Trains an `Annex.Learner` given `learner`, `data`, `labels`, and `options`.

  The `learner` should be initialized `Learner.init_learner/2` before being
  trained.

  Returns the trained `learner` along with some measure of loss or performance.
  """
  def train(%_{} = learner, dataset, options \\ []) do
    Learner.train(learner, dataset, options)
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
    learner
    |> Learner.predict(data)
    |> Data.to_flat_list()
  end
end
