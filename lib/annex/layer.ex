defmodule Annex.Layer do
  alias Annex.{Layer, Neuron, Activation}

  defstruct activation: nil,
            neurons: nil,
            inputs: nil,
            activation_deriv: nil

  def get_neurons(%Layer{neurons: neurons}), do: neurons
  def get_activation(%Layer{activation: a}), do: a
  def get_activation_deriv(%Layer{activation_deriv: ad}), do: ad
  def get_inputs(%Layer{inputs: inputs}), do: inputs

  def get_output(%Layer{} = layer) do
    layer
    |> get_neurons()
    |> Enum.map(&Neuron.get_output/1)
  end

  def new_random(neurons, size, opts \\ []) do
    {activation, activation_deriv} =
      opts
      |> Map.new()
      |> do_fetch_activations()

    %Layer{
      neurons: Enum.map(1..neurons, fn _ -> Neuron.new_random(size) end),
      activation: activation,
      activation_deriv: activation_deriv
    }
  end

  defp do_fetch_activations(%{activation: name}) when is_atom(name) do
    name
    |> Activation.by_name()
    |> do_fetch_activations()
  end

  defp do_fetch_activations(opts_map) when is_map(opts_map) do
    activation = Map.fetch!(opts_map, :activation)
    activation_deriv = Map.fetch!(opts_map, :activation_deriv)
    do_fetch_activations({activation, activation_deriv})
  end

  defp do_fetch_activations({a, a_deriv}) when is_function(a, 1) and is_function(a_deriv, 1) do
    {a, a_deriv}
  end

  def feedforward(%Layer{} = layer, inputs) do
    layer = %Layer{layer | neurons: do_activate_neurons(layer, inputs)}
    {get_output(layer), layer}
  end

  def backprop(%Layer{} = layer, total_loss_pd, loss_pds, learn_rate, activation_deriv) do
    {next_loss_pd, neurons} =
      [get_neurons(layer), List.wrap(loss_pds)]
      |> Enum.zip()
      |> Enum.map(fn {neuron, loss_pd} ->
        Neuron.backprop(neuron, total_loss_pd, loss_pd, learn_rate, activation_deriv)
      end)
      |> Enum.unzip()

    {List.flatten(next_loss_pd), %Layer{layer | neurons: neurons}}
  end

  defp do_activate_neurons(%Layer{} = layer, inputs) do
    activation = get_activation(layer)

    layer
    |> get_neurons()
    |> Enum.map(fn
      neuron ->
        Neuron.activate(neuron, inputs, activation)
    end)
  end
end
