defmodule Annex.Layer.ActivationTest do
  use ExUnit.Case

  alias Annex.{
    AnnexError,
    Layer.Activation
  }

  describe "from_name/1" do
    test "works for relu with a threshold" do
      assert %Activation{} = Activation.from_name({:relu, 0.1})
    end

    test "works for softmax" do
      assert %Activation{} = Activation.from_name(:softmax)
    end

    test "raises for unknown name" do
      assert_raise(AnnexError, fn ->
        Activation.from_name(:blep)
      end)
    end
  end

  describe "relu with threshold" do
    test "activator works" do
      activator =
        {:relu, 0.1}
        |> Activation.from_name()
        |> Activation.get_activator()

      assert is_function(activator, 1) == true
      assert activator.(0.0) == 0.1
      assert activator.(1.0) == 1.0
    end

    test "derivative works" do
      derivative =
        {:relu, 0.1}
        |> Activation.from_name()
        |> Activation.get_derivative()

      assert is_function(derivative, 1) == true
      assert derivative.(0.0) == 0.0
      assert derivative.(0.1) == 0.0
      assert derivative.(0.11) == 1.0
      assert derivative.(1.0) == 1.0
    end
  end

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

  test "softmax/1" do
    assert Activation.softmax([1.0, 2.0, 3.0]) == [
             0.09003057317038046,
             0.24472847105479767,
             0.6652409557748219
           ]
  end

  test "get_inputs" do
    act = %Activation{inputs: [2.0, 3.0, 4.0]}
    assert Activation.get_inputs(act) == [2.0, 3.0, 4.0]
  end
end
