defmodule Annex.Utils do
  @moduledoc """
  This module holds functions that are not necessarily specific to one
  of Annex's other modules.
  """

  defguard is_pos_integer(x) when is_integer(x) and x > 0

  @doc """
  Generates a float between -1.0 and 1.0
  """
  @spec random_float() :: float()
  def random_float do
    :rand.uniform() * 2.0 - 1.0
  end

  @doc """
  A strict zip function in which the two given enumerables *must* be the same size.
  """
  @spec zip([any()], [any()]) :: [{any(), any()}]
  def zip([], []) do
    []
  end

  def zip([a | a_rest], [b | b_rest]) do
    [{a, b} | zip(a_rest, b_rest)]
  end

  def zip(%Stream{} = a, b) do
    a
    |> Enum.into([])
    |> zip(b)
  end

  def zip(a, %Stream{} = b) do
    zip(a, Enum.into(b, []))
  end

  @doc """
  A strict zip-while-mapping function in which the two given enumerables *must* be the same size.
  """
  @spec zipmap([any()], [any()], any()) :: [any()]
  def zipmap([], [], _) do
    []
  end

  def zipmap([a | a_rest], [b | b_rest], func) when is_function(func, 2) do
    [func.(a, b) | zipmap(a_rest, b_rest, func)]
  end

  # def mean(items, mapper) when is_function(mapper, 1) do
  #   {counted, totaled} =
  #     Enum.reduce(items, {0, 0.0}, fn item, {count, total} ->
  #       {count + 1, mapper.(item) + total}
  #     end)

  #   totaled / counted
  # end

  def is_module?(item) do
    is_atom(item) && Code.ensure_loaded?(item)
  end

  defmacro validate(field, reason, do: expr) do
    code = Macro.to_string(expr)
    vars = Annex.Debug.get_ast_vars(expr)

    quote do
      result = unquote(expr)

      if result != true do
        error = %Annex.AnnexError{
          message: "valiation failed",
          details: [
            module: __MODULE__,
            field: unquote(field),
            reason: unquote(reason),
            variables: Keyword.take(binding(), unquote(vars)),
            code: unquote(code)
          ]
        }

        {:error, unquote(field), error}
      else
        :ok
      end
    end
  end
end
