defmodule Annex.LearnerTest do
  use ExUnit.Case

  alias Annex.{
    Layer.Sequence,
    LayerConfig,
    Learner,
    LearnerLayerMock,
    LearnerMock
  }

  defmodule FakeLearnerWithoutTrain do
    @moduledoc """
    An Annex.Learner without a train/3 implementation
    """
    use Annex.Learner

    defstruct thing: 1

    def predict(%FakeLearnerWithoutTrain{} = learner) do
      prediction = [1.0]
      {learner, prediction}
    end
  end

  defmodule FakeLearnerWithTrain do
    @moduledoc """
    An Annex.Learner with a train/3 implementation
    """
    use Annex.Learner

    defstruct thing: 1

    def predict(%FakeLearnerWithTrain{} = learner) do
      prediction = [1.0]
      {learner, prediction}
    end

    def train(learner, dataset, opts) do
      {learner, dataset, opts}
    end
  end

  import Mox

  # Make sure mocks are verified when the test exits
  setup :verify_on_exit!

  describe "is_learner?/1" do
    test "true for learner structs" do
      assert Learner.is_learner?(%FakeLearnerWithoutTrain{}) == true
      assert Learner.is_learner?(%FakeLearnerWithTrain{}) == true
    end

    test "true for learner modules" do
      assert Learner.is_learner?(FakeLearnerWithoutTrain) == true
      assert Learner.is_learner?(FakeLearnerWithTrain) == true
    end

    test "false for non-learners" do
      assert Learner.is_learner?(URI) == false
      assert Learner.is_learner?(nil) == false
      assert Learner.is_learner?(1) == false
    end
  end

  describe "predict/2" do
    test "calls a learner's predict function" do
      data = [1.0, 2.0, 3.0]
      learner = %FakeLearnerWithTrain{}

      expect(LearnerMock, :predict, fn input_learner, input_data ->
        assert input_learner == learner
        assert input_data == data
        {input_learner, input_data}
      end)

      assert LearnerMock.predict(learner, data) == {learner, data}
    end
  end

  describe "train/2" do
    test "calls train/3 with an empty opts list as arg3" do
      dataset = [
        {[1.0, 2.0, 3.0], [1.0]}
      ]

      learner = %{__struct__: Annex.LearnerMock}

      LearnerMock
      |> expect(
        :train,
        1000,
        fn input_learner, _dataset, opts ->
          assert opts == []
          {input_learner, %{error: [1.0, 0.9], loss: 0.0}}
        end
      )
      |> expect(:init_learner, fn %{__struct__: LearnerMock} = input_learner, opts ->
        assert opts == []
        input_learner
      end)

      # fake a struct
      assert Learner.train(%{__struct__: LearnerMock}, dataset) ==
               {learner, %{error: [1.0, 0.9], loss: 0.0}}
    end
  end

  describe "train/3" do
    test "turns a LayerConfig into the applicable Layer/Learner struct and calls train" do
      cfg = LayerConfig.build(LearnerLayerMock, [])

      dataset = [
        {[1.0, 2.0, 3.0], [1.0]}
      ]

      LearnerLayerMock
      |> expect(
        :train,
        1000,
        fn input_learner, _dataset, opts ->
          assert opts == []
          {input_learner, %{error: [1.0, 0.9], loss: 0.0}}
        end
      )
      |> expect(:init_layer, fn layer_config ->
        assert %LayerConfig{} = layer_config
        %{__struct__: LearnerLayerMock}
      end)
      |> expect(:init_learner, fn input_learner, _opts ->
        assert %{__struct__: LearnerLayerMock} = input_learner
        input_learner
      end)

      assert Learner.train(cfg, dataset, []) ==
               {%{__struct__: LearnerLayerMock}, %{error: [1.0, 0.9], loss: 0.0}}
    end

    @left_true_dataset [
      {[1.0, 0.0], [1.0]},
      {[0.0, 0.0], [0.0]},
      {[1.0, 1.0], [1.0]},
      {[0.0, 1.0], [0.0]}
    ]
    test "can handle {:loss_less_than, float} halt_condition option" do
      assert {%Sequence{} = seq, outputs} =
               [
                 Annex.dense(3, 2),
                 Annex.activation(:tanh),
                 Annex.dense(1, 3),
                 Annex.activation(:tanh)
               ]
               |> Annex.sequence()
               |> Learner.train(@left_true_dataset,
                 name: "LOSS_LESS_THAN TEST",
                 halt_condition: {:loss_less_than, 0.05}
               )
    end
  end

  describe "has_train?" do
    test "true for a Learner with a train/3 implementing struct" do
      assert Learner.has_train?(%FakeLearnerWithTrain{}) == true
    end

    test "true for a Learner with a train/3 implementing module" do
      assert Learner.has_train?(FakeLearnerWithTrain) == true
    end

    test "false for non-train/3 implementing struct" do
      assert Learner.has_train?(%FakeLearnerWithoutTrain{}) == false
    end

    test "false for non-train/3 implementing module" do
      assert Learner.has_train?(FakeLearnerWithoutTrain) == false
    end

    test "false for others" do
      assert Learner.has_train?(nil) == false
      assert Learner.has_train?(URI) == false
      assert Learner.has_train?(1) == false
    end
  end

  describe "init_learner/2" do
    test "calls the init_learner/2 of the learner struct's module" do
      learner = %{__struct__: LearnerLayerMock}

      expect(LearnerLayerMock, :init_learner, fn ^learner, [] ->
        learner
      end)

      assert Learner.init_learner(learner, []) == learner
    end
  end
end
