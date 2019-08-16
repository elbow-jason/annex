defmodule Annex.LayerConfig do
  @moduledoc """
  The Annex.LayerConfig is the intermediate structure used to intialize an Annex.Layer.

  This is particularly useful for building the same Layer given many different combinations
  of configuration.
  """
  alias Annex.{
    AnnexError,
    Layer,
    LayerConfig
  }

  @type details :: %{atom() => any}
  @type t(layer_module) :: %__MODULE__{
          details: details(),
          module: layer_module
        }

  defstruct details: %{},
            module: nil

  def build(module, kvs \\ []) when is_atom(module) do
    %LayerConfig{
      module: module,
      details: Map.new(kvs)
    }
  end

  @spec module(t(module())) :: module()
  def module(%LayerConfig{module: m}), do: m

  def details(%LayerConfig{details: d}), do: d

  @spec add(t(module()), atom(), any()) :: t(module())
  def add(cfg, key, value) do
    add(cfg, %{key => value})
  end

  @spec add(t(module()), any) :: t(module())
  def add(%LayerConfig{} = cfg, details) do
    more_details = Map.new(details)
    %LayerConfig{cfg | details: cfg |> details |> Map.merge(more_details)}
  end

  @spec init_layer(t(module())) :: Layer.t()
  def init_layer(%LayerConfig{} = cfg) do
    module(cfg).init_layer(cfg)
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

  def get(%LayerConfig{} = cfg, key, default \\ nil) do
    case fetch(cfg, key) do
      {:ok, _, value} -> value
      {:error, _, _} -> default
    end
  end
end
