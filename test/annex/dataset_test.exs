defmodule Annex.DatasetTest do
  use ExUnit.Case
  alias Annex.Dataset
  alias Annex.Utils

  describe "zip/2" do
    test "give two same-length lists (inputs and labels) returns a zipped list of {inputs_row, labels_row}" do
      inputs = [
        [1.0, 2.0, 3.0],
        [10.0, 11.0, 12.0]
      ]

      labels = [
        [21.0, 22.0],
        [58.0, 42.0]
      ]

      assert Dataset.zip(inputs, labels) == [
               {[1.0, 2.0, 3.0], [21.0, 22.0]},
               {[10.0, 11.0, 12.0], [58.0, 42.0]}
             ]
    end
  end

  def random_dataset(n_inputs, n_labels, n_rows) do
    fn ->
      inputs = Utils.random_weights(n_inputs)
      labels = Utils.random_weights(n_labels)
      {inputs, labels}
    end
    |> Stream.repeatedly()
    |> Enum.take(n_rows)
  end

  describe "randomize/1" do
    test "works" do
      # this test may fail occasionally.
      dataset = random_dataset(2, 2, 50)
      dataset_size = length(dataset)
      assert dataset_size == 50
      randomized = Dataset.randomize(dataset)
      assert dataset != randomized
      assert length(randomized) == dataset_size
      assert MapSet.new(dataset) == MapSet.new(randomized)
    end
  end

  describe "split/2" do
    test "works" do
      dataset = random_dataset(2, 2, 50)

      {left, right} = Dataset.split(dataset, 0.3)
      assert length(left) == 15
      assert length(right) == 35
    end
  end
end
