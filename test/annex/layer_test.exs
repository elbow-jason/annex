defmodule Annex.LayerTest do
  use Annex.LayerCase

  alias Annex.{
    Layer,
    Layer.Dense
  }

  defmodule LayerImplementer do
    use Annex.Layer

    defstruct some_field: nil

    def init_layer(%LayerConfig{}) do
      %LayerImplementer{}
    end

    def feedforward(%LayerImplementer{} = layer, inputs) do
      {layer, inputs}
    end

    def backprop(%LayerImplementer{} = layer, error, backprops) do
      {layer, error, backprops}
    end
  end

  defmodule LayerNonImplementer do
    defstruct thing: nil
  end

  def dense_fixture do
    weights = [-0.3333, 0.24, 0.1, 0.7, -0.4, -0.9]
    biases = [1.0, 1.0]
    build(Dense, rows: 2, columns: 3, weights: weights, biases: biases)
  end

  describe "is_layer?/1" do
    test "is true for a Layer-using module" do
      assert Layer.is_layer?(%LayerImplementer{}) == true
      assert Layer.is_layer?(LayerImplementer) == true
    end

    test "is false for a non-Layer-using module" do
      assert Layer.is_layer?(%LayerNonImplementer{}) == false
      assert Layer.is_layer?(LayerNonImplementer) == false
    end

    test "is false for a non-modules" do
      assert Layer.is_layer?(:beef) == false
      assert Layer.is_layer?('carrot') == false
      assert Layer.is_layer?("cake") == false
      assert Layer.is_layer?(1) == false
      assert Layer.is_layer?(1.0) == false
      assert Layer.is_layer?(nil) == false
    end
  end

  describe "forward_shape/1" do
    test "given a shaped layer returns the layer's columns as rows and :any as the columns" do
      layer = dense_fixture()
      assert {inputs, _outputs} = Layer.shapes(layer)
      assert [rows, columns] = inputs
      assert rows == 2
      assert columns == 3
      assert Layer.forward_shape(layer) == [columns, :any]
    end
  end

  describe "backward_shape/1" do
    test "given a shaped layer returns the layer's rows as columns and :any as the rows" do
      layer = dense_fixture()
      assert {inputs, outputs} = Layer.shapes(layer)
      assert [rows, columns] = inputs
      assert rows == 2
      assert columns == 3
      assert Layer.backward_shape(layer) == [:any, rows]
    end
  end
end
