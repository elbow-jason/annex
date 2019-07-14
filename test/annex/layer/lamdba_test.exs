defmodule Annex.Layer.LambdaTest do
  use ExUnit.Case, async: true

  alias Annex.{
    Layer.Backprop,
    Layer.Lambda
  }

  defp sender(pid, message) do
    do_send = fn args, output ->
      _ = send(pid, {message, args, output})
      output
    end

    # on_encoded? = fn lambda, data ->
    #   do_send.([lambda, data], false)
    # end

    # on_encode = fn lambda, data ->
    #   do_send.([lambda, data], data)
    # end

    on_shapes = fn lambda ->
      do_send.([lambda], {:any, :any})
    end

    on_init_layer = fn lambda, opts ->
      do_send.([lambda, opts], {:ok, lambda})
    end

    on_feedforward = fn lambda, input ->
      do_send.([lambda, input], {lambda, input})
    end

    on_backprop = fn lambda, error, backprops ->
      do_send.([lambda, error, backprops], {lambda, error, backprops})
    end

    %Lambda{
      on_shapes: on_shapes,
      on_init_layer: on_init_layer,
      on_feedforward: on_feedforward,
      on_backprop: on_backprop
    }
  end

  defp assert_no_receive(timeout) do
    receive do
      x ->
        raise "Expected no receive but got #{inspect(x)}"
    after
      timeout ->
        :ok
    end
  end

  setup do
    sender = sender(self(), :yes)
    nils = %Lambda{}
    {:ok, sender: sender, nils: nils}
  end

  describe "on_shapes" do
    test "works with functions", %{sender: sender} do
      data = [1.0, 2.0, 3.0]
      assert Lambda.shapes(sender) == {:any, :any}
      assert_receive({:yes, [^sender], {:any, :any}}, 50)
    end
  end

  describe "on_init_layer" do
    test "works with functions", %{sender: sender} do
      opts = [blep: true, blop: false]
      assert Lambda.init_layer(sender, opts) == {:ok, sender}
      assert_receive({:yes, [^sender, ^opts], {:ok, ^sender}}, 50)
    end

    test "works with nil", %{nils: nils} do
      opts = [blep: true, blop: false]
      assert Lambda.init_layer(nils, opts) == {:ok, nils}
      assert_no_receive(50)
    end
  end

  describe "on_feedforward" do
    test "works for functions", %{sender: sender} do
      input = [1.1, 2.2, 3.3]
      assert Lambda.feedforward(sender, input) == {sender, input}
      assert_receive({:yes, [^sender, ^input], {sender, ^input}}, 50)
    end

    test "works for nil", %{nils: nils} do
      input = [1.1, 2.2, 3.3]
      assert Lambda.feedforward(nils, input) == {nils, input}
      assert_no_receive(50)
    end
  end

  describe "on_backprop" do
    test "works for functions", %{sender: sender} do
      error = [1.1, 1.2, 1.3, 1.4]
      props = Backprop.new()
      return = {sender, error, props}
      assert Lambda.backprop(sender, error, props) == return
      assert_receive({:yes, [^sender, ^error, ^props], ^return}, 50)
    end

    test "works for nil", %{nils: nils} do
      error = [1.1, 1.2, 1.3, 1.4]
      props = Backprop.new()
      return = {nils, error, props}
      assert Lambda.backprop(nils, error, props) == return
      assert_no_receive(50)
    end
  end

  describe "state functions" do
    test "work together", %{nils: nils} do
      assert Lambda.get_state(nils) == nil
      nils = Lambda.put_state(nils, "yup")
      assert Lambda.get_state(nils) == "yup"
      nils = Lambda.update_state(nils, fn state -> state <> " nope" end)
      assert Lambda.get_state(nils) == "yup nope"
    end
  end
end
