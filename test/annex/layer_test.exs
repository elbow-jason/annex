defmodule Annex.LayerTest do
  use ExUnit.Case

  alias Annex.{
    Layer,
    Layer.Dense
  }

  def dense_fixture do
    weights = [-0.3333, 0.24, 0.1, 0.7, -0.4, -0.9]
    biases = [1.0, 1.0]
    Dense.build(2, 3, weights, biases)
  end

  describe "forward_shape/1" do
    test "given a shaped layer returns the layer's columns as rows and :any as the columns" do
      layer = dense_fixture()
      assert {rows, columns} = Layer.shape(layer)
      assert rows == 2
      assert columns == 3
      assert Layer.forward_shape(layer) == {columns, :any}
    end
  end

  describe "backward_shape/1" do
    test "given a shaped layer returns the layer's rows as columns and :any as the rows" do
      layer = dense_fixture()
      assert {rows, columns} = Layer.shape(layer)
      assert rows == 2
      assert columns == 3
      assert Layer.backward_shape(layer) == {:any, rows}
    end
  end
end
