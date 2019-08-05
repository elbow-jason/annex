defmodule Annex.ShapeTest do
  use ExUnit.Case

  alias Annex.{
    AnnexError,
    Shape
  }

  describe "convert_abstract_to_concrete/2" do
    test "when abstract is _actually_ a concrete returns the abstract when compatible" do
      abstract_shape = [12, 2]
      assert Shape.kind(abstract_shape) == :concrete
      assert Shape.convert_abstract_to_concrete(abstract_shape, [2, 3, 4]) == [12, 2]
    end

    test "when abstract is _actually_ a concrete raises when incompatible" do
      abstract_shape = [13, 2]
      assert Shape.kind(abstract_shape) == :concrete

      assert_raise(AnnexError, fn ->
        Shape.convert_abstract_to_concrete(abstract_shape, [2, 3, 4])
      end)
    end

    test "when abstract shape has :any as the first element a compatible concrete shape" do
      abstract = [:any, 3]
      assert Shape.kind(abstract) == :abstract
      concrete = [2, 2, 3]
      assert Shape.convert_abstract_to_concrete(abstract, concrete) == [4, 3]
    end

    test "when abstract shape is abstract returns a compatible concrete shape" do
      abstract = [4, :any]
      assert Shape.kind(abstract) == :abstract
      concrete = [2, 2, 3]
      assert Shape.convert_abstract_to_concrete(abstract, concrete) == [4, 3]
    end

    test "when abstract shape is abstract returns a compatible concrete shape 2" do
      abstract = [4, :any]
      assert Shape.kind(abstract) == :abstract
      concrete = [2, 2, 2, 3]
      assert Shape.convert_abstract_to_concrete(abstract, concrete) == [4, 6]
    end

    test "raises for incompatible shapes" do
      abstract = [4, :any]
      assert Shape.kind(abstract) == :abstract
      concrete = [3, 3]

      assert_raise(AnnexError, fn ->
        Shape.convert_abstract_to_concrete(abstract, concrete)
      end)
    end

    test "works for compatible concrete shapes" do
      concrete1 = [2, 2, 3]
      concrete2 = [4, 3]
      assert Shape.kind(concrete1) == :concrete
      assert Shape.kind(concrete2) == :concrete
      assert Shape.convert_abstract_to_concrete(concrete1, concrete2) == [2, 2, 3]
    end

    test "raises for incompatible concrete shapes" do
      concrete1 = [5, 3]
      concrete2 = [4, 3]
      assert Shape.kind(concrete1) == :concrete
      assert Shape.kind(concrete2) == :concrete

      exception =
        assert_raise(AnnexError, fn ->
          Shape.convert_abstract_to_concrete(concrete1, concrete2)
        end)

      assert exception == %AnnexError{
               details: [
                 reason: "abstract_size was larger than concrete_size",
                 concrete_size: 12,
                 abstract_size: 15,
                 abstract: [5, 3],
                 target_concrete: [4, 3]
               ],
               message:
                 "Annex.Shape encountered an error while turning an abstract shape to a concrete shape."
             }
    end
  end

  describe "is_factor_of?/2" do
    test "true for ints that are factors" do
      assert Shape.is_factor_of?(16, 4) == true
      assert Shape.is_factor_of?(12, 4) == true
      assert Shape.is_factor_of?(8, 4) == true
      assert Shape.is_factor_of?(4, 4) == true
    end

    test "false for non-factors" do
      assert Shape.is_factor_of?(17, 4) == false
      assert Shape.is_factor_of?(15, 4) == false
      assert Shape.is_factor_of?(9, 4) == false
      assert Shape.is_factor_of?(5, 4) == false
    end
  end

  describe "is_shape?/1" do
    test "true for concrete shape - list of pos_integer(s)" do
      assert Shape.is_shape?([2, 3, 4]) == true
    end

    test "true for abstract shapes - list of pos_integer and :any(s)" do
      assert Shape.is_shape?([2, :any, 4]) == true
    end

    test "false for non-shape-likes" do
      assert Shape.is_shape?(:defer) == false
      assert Shape.is_shape?(-1) == false
      assert Shape.is_shape?(0) == false
      assert Shape.is_shape?(nil) == false
      assert Shape.is_shape?(1.1) == false
      assert Shape.is_shape?(:other) == false
      assert Shape.is_shape?([]) == false
      assert Shape.is_shape?(false) == false
      assert Shape.is_shape?(true) == false
      assert Shape.is_shape?(%{}) == false

      assert Shape.is_shape?({-1}) == false
      assert Shape.is_shape?({0}) == false
      assert Shape.is_shape?({nil}) == false
      assert Shape.is_shape?({1.1}) == false
      assert Shape.is_shape?({:other}) == false
      assert Shape.is_shape?({'thing'}) == false
      assert Shape.is_shape?({[]}) == false
      assert Shape.is_shape?({false}) == false
      assert Shape.is_shape?({true}) == false
      assert Shape.is_shape?({%{}}) == false
    end
  end

  describe "is_shape_value?/1" do
    test "true for positive integer" do
      assert Shape.is_shape_value?(1) == true
    end

    test "false for 0" do
      assert Shape.is_shape_value?(0) == false
    end

    test "false for negative integer" do
      assert Shape.is_shape_value?(-1) == false
    end

    test "true for :any" do
      assert Shape.is_shape_value?(:any) == true
    end

    test "false for non-positive-integer or non-:any" do
      assert Shape.is_shape_value?(-1) == false
      assert Shape.is_shape_value?(0) == false
      assert Shape.is_shape_value?(nil) == false
      assert Shape.is_shape_value?(1.1) == false
      assert Shape.is_shape_value?(:other) == false
      assert Shape.is_shape_value?('thing') == false
      assert Shape.is_shape_value?([]) == false
      assert Shape.is_shape_value?(false) == false
      assert Shape.is_shape_value?(true) == false
      assert Shape.is_shape_value?(%{}) == false
    end
  end

  describe "product/1" do
    test "works for concrete shapes" do
      assert Shape.product([2, 3, 4]) == 2 * 3 * 4
    end

    test "raises ArithmeticError for abstract shapes" do
      assert_raise(ArithmeticError, fn -> Shape.product([2, :any, 4]) end)
    end
  end

  describe "factor/1" do
    test "for a concrete is the same as product" do
      factor = Shape.factor([2, 3, 4])
      assert is_integer(factor) == true
      assert factor == Shape.product([2, 3, 4])
      assert factor == 2 * 3 * 4
    end

    test "for an abstract returns the product of the integers" do
      factor = Shape.factor([2, :any, 4])
      assert is_integer(factor) == true
      assert factor == 2 * 4
    end
  end

  describe "kind/1" do
    test "returns :concrete for tuple of integers" do
      assert Shape.kind([1, 2, 4]) == :concrete
    end

    test "returns :abstract for tuple with integers and some :any atoms" do
      assert Shape.kind([2, :any, 3]) == :abstract
    end
  end
end
