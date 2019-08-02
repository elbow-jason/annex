defmodule Annex.LayerConfig do
  alias Annex.{
    AnnexError,
    LayerConfig
  }

  @type details :: %{atom() => any}
  @type t :: %__MODULE__{
          details: details(),
          module: module()
        }

  defstruct details: %{},
            module: nil

  def build(module, kvs \\ []) when is_atom(module) do
    %LayerConfig{
      module: module,
      details: Map.new(kvs)
    }
  end

  @spec module(t()) :: module()
  def module(%LayerConfig{module: m}), do: m

  def details(%LayerConfig{details: d}), do: d

  @spec add(t(), atom(), any()) :: t()
  def add(cfg, key, value) do
    add(cfg, %{key => value})
  end

  @spec add(t(), any) :: t()
  def add(%LayerConfig{} = cfg, details) do
    more_details = Map.new(details)
    %LayerConfig{cfg | details: cfg |> details |> Map.merge(more_details)}
  end

  def build_layer(%LayerConfig{} = cfg) do
    module(cfg).build(cfg)
  end

  def fetch(%LayerConfig{} = cfg, key) do
    cfg
    |> details()
    |> Map.fetch(key)
    |> case do
      {:ok, value} ->
        {:ok, key, value}

      :error ->
        error = %AnnexError{
          message: "is required",
          details: [
            key: key,
            layer_module: module(cfg)
          ]
        }

        {:error, key, error}
    end
  end

  def fetch_lazy(%LayerConfig{} = cfg, key, func) when is_function(func, 0) do
    case fetch(cfg, key) do
      {:ok, _, _} = found ->
        found

      {:error, key, _} ->
        {:ok, key, func.()}
    end
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
            reason: unquote(reason),
            code: unquote(code),
            variables: Keyword.take(binding(), unquote(vars))
          ]
        }

        {:error, unquote(field), error}
      else
        :ok
      end
    end
  end
end
