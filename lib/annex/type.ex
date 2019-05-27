defmodule Annex.Type do
  defguard is_list_of_floats(floats) when is_float(hd(floats))
end
