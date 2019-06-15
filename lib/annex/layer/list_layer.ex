defmodule Annex.Layer.ListLayer do
  @type t :: [float(), ...]

  defmacro __using__(_) do
    quote do
      @before_compile Annex.Layer.ListLayer
    end
  end

  defmacro __before_compile__(_env) do
    quote do
      alias Annex.Layer.ListLayer

      if not Module.defines?(__MODULE__, {:encode, 2}) do
        defdelegate encode(layer, data), to: ListLayer
      end

      if not Module.defines?(__MODULE__, {:decode, 2}) do
        defdelegate decode(layer, data), to: ListLayer
      end

      if not Module.defines?(__MODULE__, {:encoded?, 2}) do
        defdelegate encoded?(layer, data), to: ListLayer
      end
    end
  end

  def encode(%_{}, [f | _] = data) when is_float(f) do
    data
  end

  def encode(%_{}, enumerable) do
    enumerable
    |> Enum.to_list()
    |> List.flatten()
  end

  def encoded?(%_{}, [f | _]), do: is_float(f)
  def encoded?(%_{}, _), do: false

  def decode(%_{}, [f | _] = data) when is_float(f), do: data
  def decode(%_{}, data) when is_list(data), do: List.flatten(data)
end
