defmodule Annex.Layer.ListLayer do
  defmacro __using__(_) do
    quote do
      @before_compile Annex.Layer.ListLayer
    end
  end

  defmacro __before_compile__(_env) do
    quote do
      if not Module.defines?(__MODULE__, {:encode, 2}) do
        def encode(%_{}, [f | _] = data) when is_float(f) do
          data
        end

        def encode(%_{}, enumerable) do
          enumerable
          |> Enum.to_list()
          |> List.flatten()
        end
      end

      if not Module.defines?(__MODULE__, {:decode, 2}) do
        def decode(%_{}, [f | _] = data) do
          data
        end

        def decode(%_{}, data) when is_list(data) do
          List.flatten(data)
        end
      end

      if not Module.defines?(__MODULE__, {:encoded?, 2}) do
        def encoded?(%_{}, [f | _]), do: is_float(f)
        def encoded?(%_{}, _), do: false
      end
    end
  end
end
