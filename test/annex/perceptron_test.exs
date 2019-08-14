defmodule Annex.PerceptronTest do
  use ExUnit.Case

  alias Annex.Layer.Activation
  alias Annex.Perceptron

  def sigmoid(x), do: Activation.sigmoid(x)

  describe "new/3" do
    test "returns a Perceptron struct" do
      activation = &sigmoid/1

      assert %Perceptron{
               weights: weights,
               bias: 1.0,
               activation: ^activation,
               learning_rate: 0.05
             } = Perceptron.new(6, activation)

      assert Enum.all?(weights, &is_float/1) == true
    end

    test "weights with the same legnth as input will make into the struct from the options" do
      weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      assert %Perceptron{weights: ^weights} = Perceptron.new(6, &sigmoid/1, weights: weights)
    end
  end

  describe "predict/2" do
    test "works" do
      inputs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      perceptron = Perceptron.new(6, &sigmoid/1, [])
      prediction = Perceptron.predict(perceptron, inputs)
      assert is_float(prediction) == true
    end
  end

  describe "train/3" do
    test "works for AND" do
      dataset = [
        {[1.0, 1.0, 1.0], 1.0},
        {[1.0, 0.0, 1.0], 0.0},
        {[0.0, 0.0, 0.0], 0.0},
        {[1.0, 0.0, 0.0], 0.0},
        {[0.0, 0.0, 1.0], 0.0},
        {[0.0, 1.0, 0.0], 0.0},
        {[0.0, 1.0, 1.0], 0.0}
      ]

      p1 = Perceptron.new(3, &sigmoid/1, [])
      p2 = Perceptron.train(p1, dataset, runs: 30_000)
      assert %Perceptron{} = p1
      assert %Perceptron{} = p2
      one = Perceptron.predict(p2, [1.0, 1.0, 1.0])
      zero = Perceptron.predict(p2, [0.0, 1.0, 1.0])
      assert_in_delta(one, 1.0, 0.1)
      assert_in_delta(zero, 0.0, 0.1)
    end
  end
end
