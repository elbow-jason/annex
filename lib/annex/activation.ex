defmodule Annex.Activation do
  def by_name(key) do
    case key do
      {:relu, threshold} ->
        activation = fn n -> relu_with_threshold(n, threshold) end
        activation_deriv = fn n -> relu_deriv(n, threshold) end
        {activation, activation_deriv}

      :relu ->
        {&relu/1, &relu_deriv/1}

      :sigmoid ->
        {&sigmoid/1, &sigmoid_deriv/1}

      :tanh ->
        {&tanh/1, &tanh_deriv/1}
    end
  end

  def relu(n) do
    relu_with_threshold(n, 0.0)
  end

  def relu_deriv(x), do: relu_deriv(x, 0.0)
  def relu_deriv(x, threshold) when x > threshold, do: 1.0
  def relu_deriv(_, _), do: 0.0

  def relu_with_threshold(n, threshold) do
    if n > threshold do
      n
    else
      threshold
    end
  end

  def sigmoid(n) do
    1.0 / (1.0 + :math.exp(-n))
  end

  def sigmoid_deriv(x) do
    fx = sigmoid(x)
    fx * (1 - fx)
  end

  def tanh(n) do
    :math.tanh(n)
  end

  def tanh_deriv(x) do
    tanh_squared =
      x
      |> :math.tanh()
      |> :math.pow(2)

    1.0 - tanh_squared
  end
end
