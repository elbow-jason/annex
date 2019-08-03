defmodule Annex.AnnexError do
  alias Annex.AnnexError

  @type t :: %__MODULE__{
          message: String.t(),
          details: Keyword.t()
        }

  defexception [:message, details: []]

  @spec build(String.t(), Keyword.t()) :: t()
  def build(message, details) do
    %AnnexError{
      message: message,
      details: details
    }
  end

  @spec add_details(t(), Keyword.t()) :: t()
  def add_details(%AnnexError{details: prev} = error, details) when is_list(details) do
    %AnnexError{error | details: prev ++ details}
  end

  @spec message(t()) :: String.t()
  def message(%AnnexError{message: message, details: details}) do
    """
    #{message}
    #{render_details(details)}
    """
  end

  defp render_details(details) do
    details
    |> Enum.map(fn
      {key, [{subkey, _} | _] = kwargs} when is_atom(subkey) ->
        value =
          kwargs
          |> Enum.map(fn {k, v} ->
            "  #{k}: #{inspect(v)}"
          end)
          |> Enum.intersperse("\n")
          |> IO.iodata_to_binary()

        "#{key}: [\n#{value}\n]"

      {:code, code} ->
        "code: #{code}"

      {key, value} ->
        "#{key}: #{inspect(value)}"
    end)
    |> Enum.intersperse("\n")
    |> IO.iodata_to_binary()
  end
end
