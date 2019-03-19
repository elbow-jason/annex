defmodule Annex.Layer do
  @callback backprop(struct(), float(), [float(), ...], Keyword.t()) :: {[float(), ...], struct}
  @callback feedforward(struct(), [float(), ...]) :: {[float(), ...], struct()}

  alias Annex.{Layer, Neuron, Activator}

  defstruct activator: nil,
            neurons: nil,
            inputs: nil,
            activation_deriv: nil

  def get_neurons(%Layer{neurons: neurons}), do: neurons

  def get_activation(%Layer{activator: %Activator{} = a}) do
    Activator.get_activation(a)
  end

  def get_activation_deriv(%Layer{activator: %Activator{} = a}) do
    Activator.get_derivative(a)
  end

  def get_inputs(%Layer{inputs: inputs}), do: inputs

  def get_output(%Layer{} = layer) do
    layer
    |> get_neurons()
    |> Enum.map(&Neuron.get_output/1)
  end

  def new_random(neurons, size, %Activator{} = activator) do
    %Layer{
      neurons: Enum.map(1..neurons, fn _ -> Neuron.new_random(size) end),
      activator: activator
    }
  end

  def new_random(neurons, size, name) when is_atom(name) do
    new_random(neurons, size, Activator.build(name))
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
    activation =
      layer
      |> get_activation()

    layer
    |> get_neurons()
    |> Enum.map(fn
      neuron ->
        Neuron.activate(neuron, inputs, activation)
    end)
  end
end
