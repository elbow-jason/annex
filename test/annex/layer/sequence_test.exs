defmodule Annex.Layer.SequenceTest do
  use Annex.LayerCase

  alias Annex.Layer
  alias Annex.LayerConfig

  alias Annex.Layer.{
    Activation,
    Backprop,
    Dense,
    Dropout,
    Sequence
  }

  alias Annex.Data.DMatrix

  def generate_n_layers(n) when rem(n, 3) == 0 or n < 3 do
    # the n_layers should be a multiple of n layers.
    # else wierd configuration issues happen.
    [
      Annex.dense(1, 1),
      Annex.activation(:sigmoid),
      Annex.dropout(0.5)
    ]
    |> Stream.cycle()
    |> Enum.take(n)
  end

  def generate_sequence(layer_configs) do
    assert {:ok, seq} =
             Sequence
             |> LayerConfig.build(layers: layer_configs)
             |> LayerConfig.init_layer()

    seq
  end

  def in_order?([Dense, Activation | _rest]), do: true
  def in_order?([Activation, Dropout | _rest]), do: true
  def in_order?([Dropout, Dense | _rest]), do: true
  def in_order?(_), do: false

  def all_in_order?(layers) do
    layers
    |> to_module()
    |> do_all_in_order?()
  end

  defp do_all_in_order?([_]), do: true

  defp do_all_in_order?(layers) do
    if in_order?(layers) do
      layers
      |> tl()
      |> all_in_order?()
    else
      false
    end
  end

  defp to_module(list) when is_list(list), do: Enum.map(list, &to_module/1)
  defp to_module(%LayerConfig{} = cfg), do: LayerConfig.module(cfg)
  defp to_module(%module{}), do: module
  defp to_module(module) when is_atom(module), do: module

  def assert_in_order(%Sequence{layers: layers}) do
    assert_in_order(layers)
  end

  def assert_in_order(layers) when is_map(layers) do
    assert MapArray.to_list(layers)
  end

  def assert_in_order(layers) when is_list(layers) do
    assert all_in_order?(layers), "Layers were not in order: #{inspect(layers)}"
  end

  def run_order_assertions(n, seq_transform) do
    layer_configs = generate_n_layers(n)

    assert %Sequence{} = seq = generate_sequence(layer_configs)

    # make sure seq is in order
    seq_layers = Sequence.get_layers(seq)
    assert MapArray.len(seq_layers) == n
    layers = MapArray.to_list(seq_layers)
    assert_in_order(layers)

    # apply transform
    assert {:ok, %Sequence{} = seq} =
             layer_configs
             |> seq_transform.()
             |> Sequence.init_layer()

    # make sure seq is in order
    seq_layers = Sequence.get_layers(seq)
    assert MapArray.len(seq_layers) == n
    layers = MapArray.to_list(seq_layers)
    assert_in_order(layers)
  end

  setup do
    layer_configs = generate_n_layers(9)
    seq = generate_sequence(layer_configs)
    {:ok, %{seq: seq, layer_configs: layer_configs}}
  end

  describe "layer ordering:" do
    test "order helpers work" do
      layers = generate_n_layers(9)
      assert all_in_order?(layers)
      refute layers |> Enum.reverse() |> all_in_order?
    end

    test "Annex.sequence/2 preserves ordering of 2 layers" do
      run_order_assertions(2, fn layers ->
        Annex.sequence(layers)
      end)
    end

    test "Annex.sequence/2 preserves ordering of 3 layers" do
      run_order_assertions(3, fn layers ->
        Annex.sequence(layers)
      end)
    end

    test "Annex.sequence/2 preserves ordering of 12 layers" do
      run_order_assertions(12, fn layers ->
        Annex.sequence(layers)
      end)
    end

    test "Sequence.init_layer/1 preserves ordering of layers", %{layer_configs: layer_configs} do
      seq_cfg = LayerConfig.build(Sequence, layers: layer_configs)
      assert_in_order(layer_configs)
      assert {:ok, seq} = Sequence.init_layer(seq_cfg)
      assert_in_order(seq)
    end

    test "Sequence.feedforward/2 preserves ordering of layers", %{seq: seq1} do
      assert {%Sequence{} = seq2, _} = Sequence.feedforward(seq1, [1.0])
      assert_in_order(seq1)
      assert_in_order(seq2)
    end

    test "Sequence.backprop/2 preserves ordering of layers", %{seq: seq1} do
      assert {%Sequence{} = seq2, _} = Sequence.feedforward(seq1, [1.0])
      props = Backprop.new(negative_gradient: 0.1)
      assert {%Sequence{} = seq3, _, _} = Sequence.backprop(seq2, [1.0], props)
      assert_in_order(seq1)
      assert_in_order(seq2)
      assert_in_order(seq3)
    end
  end

  describe "feedforward and backprop works for simple sequence" do
    setup do
      dense =
        LayerConfig.build(Dense,
          rows: 2,
          columns: 3,
          weights: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          biases: [1.0, 1.0]
        )

      assert {:ok, seq} =
               Sequence
               |> LayerConfig.build(layers: [dense])
               |> Layer.init_layer()

      {:ok, seq: seq}
    end

    test "works", %{seq: seq1} do
      assert %Sequence{} = seq1
      {seq2, pred} = Sequence.feedforward(seq1, [1.0, 2.0, 3.0])
      assert pred == DMatrix.build([[15.0], [33.0]])
      input = DMatrix.build([[1.0], [2.0], [3.0]])
      %{0 => dense1} = Sequence.get_layers(seq1)
      %{0 => dense2} = Sequence.get_layers(seq2)

      assert %Dense{
               dense1
               | input: input,
                 output: pred
             } == dense2
    end
  end
end
