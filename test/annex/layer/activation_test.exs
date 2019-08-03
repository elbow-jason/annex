defmodule Annex.Layer.ActivationTest do
  use ExUnit.Case

  alias Annex.Layer.Activation

  test "relu/1" do
    assert Activation.relu(1.0) == 1.0
    assert Activation.relu(0.5) == 0.5
    assert Activation.relu(0.0) == 0.0
    assert Activation.relu(-10.0) == 0.0
  end

  test "relu_deriv/1" do
    assert Activation.relu_deriv(10.0) == 1.0
    assert Activation.relu_deriv(1.0) == 1.0
    assert Activation.relu_deriv(0.1) == 1.0
    assert Activation.relu_deriv(0.0) == 0.0
    assert Activation.relu_deriv(-1.0) == 0.0
    assert Activation.relu_deriv(-0.0001) == 0.0
  end

  test "sigmoid/1" do
    assert Activation.sigmoid(1.0) == 0.7310585786300049
    assert Activation.sigmoid(0.0) == 0.5
    assert Activation.sigmoid(-1.0) == 0.2689414213699951
  end

  test "sigmoid_deriv/1" do
    assert Activation.sigmoid_deriv(2.0) == 0.10499358540350662
    assert Activation.sigmoid_deriv(1.0) == 0.19661193324148185
    assert Activation.sigmoid_deriv(0.0) == 0.25
    assert Activation.sigmoid_deriv(-1.0) == 0.19661193324148185
  end

  test "tanh/1" do
    assert Activation.tanh(1.0) == 0.7615941559557649
    assert Activation.tanh(0.0) == 0.0
    assert Activation.tanh(-1.0) == -0.7615941559557649
  end

  test "tanh_deriv/1" do
    assert Activation.tanh_deriv(1.0) == 0.41997434161402614
    assert Activation.tanh_deriv(0.0) == 1.0
    assert Activation.tanh_deriv(-1.0) == 0.41997434161402614
  end
end
