defmodule Annex do
  alias Annex.{
    Layer.Activation,
    Layer.Dense,
    Layer.Dropout,
    Layer.Sequence,
    Learner
  }

  @spec sequence(list(struct()), keyword()) :: Sequence.t()
  def sequence(layers, opts \\ []) when is_list(layers) do
    Sequence.build([{:layers, layers} | opts])
  end

  @spec dropout(float()) :: Dropout.t()
  def dropout(frequency) do
    Dropout.build(frequency)
  end

  @spec initialize(struct()) :: struct()
  def initialize(%module{} = layer) do
    module.initialize(layer)
  end

  @spec dense(pos_integer(), keyword()) :: Dense.t()
  def dense(rows, opts \\ []) do
    cols = Keyword.get(opts, :input_dims)
    weights = Keyword.get(opts, :weights)
    biases = Keyword.get(opts, :biases)

    if is_list(weights) and is_list(biases) do
      Dense.build_from_data(rows, cols, weights, biases)
    else
      Dense.build_random(rows, cols)
    end
  end

  @spec activation(Activation.func_name()) :: Activation.t()
  def activation(name) do
    Activation.build(name)
  end

  @doc """
  Trains the given `Annex.Learner` given the `learner`, `data`, `labels`, and `options`.
  """
  def train(%_{} = learner, data, labels, options \\ []) do
    Learner.train(learner, data, labels, options)
  end

  @spec predict(struct(), any()) :: any()
  def predict(learner, data) do
    Learner.predict(learner, data)
  end
end
