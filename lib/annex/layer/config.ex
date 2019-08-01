defmodule Layer.Configuration do
  alias Layer.Configuration

  @type details :: %{atom() => any}
  @type t :: %__MODULE__{
          details: details(),
          module: module()
        }

  defstruct details: %{},
            module: nil

  def build(module, kvs \\ []) when is_atom(module) do
    %Configuration{
      module: module,
      details: Map.new(kvs)
    }
  end

  @spec module(t()) :: module()
  def module(%Configuration{module: m}), do: m

  def details(%Configuration{details: d}), do: d

  @spec add(t(), atom(), any()) :: t()
  def add(cfg, key, value) do
    add(cfg, %{key => value})
  end

  @spec add(t(), any) :: t()
  def add(%Configuration{} = cfg, details) do
    more_details = Map.new(details)
    %Configuration{cfg | details: cfg |> details |> Map.merge(more_details)}
  end

  def build_layer(%Configuration{} = cfg) do
    module = module(cfg)
    details = details(cfg)
    module.build(details)
  end
end
