defmodule Annex.Debug do
  @moduledoc """
  A module for running checks and getting helpful messages.
  """

  defmacro __using__(opts \\ []) do
    quote do
      opts = unquote(opts)
      debug_was_passed? = Keyword.get(opts, :debug, false)

      debug_was_configured? =
        Application.get_env(:annex, __MODULE__, [])
        |> Keyword.get(:debug, false)

      annex_debug = debug_was_passed? || debug_was_configured? || false
      :ok = Module.put_attribute(__MODULE__, :annex_debug, annex_debug)
      import Annex.Debug, only: [debug: 1, debug_assert: 2]
    end
  end

  defmacro debug_assert(reason, do: expr) do
    if Module.get_attribute(__CALLER__.module, :annex_debug, false) do
      code = Macro.to_string(expr)
      vars = Annex.Debug.get_ast_vars(expr)

      quote do
        result = unquote(expr)

        if result != true do
          reason = unquote(reason)
          vars = unquote(vars)
          code = unquote(code)
          variables = Keyword.take(binding(), vars)

          message = """
          Debug assertion failed! -

          reason:
            #{inspect(reason)}

          where:
            #{Annex.Debug.format_variables(variables)}

          code:
            #{Annex.Debug.format_code(code)}

          expected:
            true

          got:
            #{inspect(result)}
          """

          raise Annex.AnnexError, message: message
        end
      end
    end
  end

  defmacro debug(do: block) do
    if Module.get_attribute(__CALLER__.module, :annex_debug, false) do
      block
    else
      :ok
    end
  end

  @doc false
  def get_ast_vars(ast) do
    do_get_ast_vars(ast, [])
  end

  defp do_get_ast_vars(ast, acc) do
    case ast do
      {name, _, atom} when is_atom(atom) ->
        [name | acc]

      {_, _, list} when is_list(list) ->
        Enum.reduce(list, acc, fn item, more_acc ->
          do_get_ast_vars(item, more_acc)
        end)

      _ ->
        acc
    end
  end

  @doc false
  def format_code(code) do
    code
    |> String.split("\n")
    |> case do
      [^code] ->
        code

      lines ->
        lines
        |> Enum.reject(fn l ->
          String.trim(l) in ["(", ")"]
        end)
        |> Enum.join("\n")
        |> String.trim()
    end
  end

  def format_variables(keywords) do
    keywords
    |> Enum.map(fn {k, v} -> "  #{k} = #{inspect(v, limit: 5)}" end)
    |> Enum.join("\n")
    |> String.trim()
  end
end
