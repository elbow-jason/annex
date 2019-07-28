defmodule Annex.AnnexError do
  alias Annex.AnnexError

  defexception [:message, details: []]

  def build(message, details) do
    %AnnexError{
      message: message,
      details: details
    }
  end

  def add_details(%AnnexError{details: prev} = error, details) do
    %AnnexError{error | details: prev ++ List.wrap(details)}
  end

  def message(%AnnexError{message: message, details: details}) do
    """
    #{message}
    #{render_details(details)}
    """
  end

  defp render_details(details) do
    details
    |> Enum.map(fn {key, value} ->
      "#{key}: #{inspect(value)}"
    end)
    |> Enum.intersperse("\n")
    |> IO.iodata_to_binary()
  end
end
