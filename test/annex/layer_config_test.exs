defmodule Annex.LayerConfigTest do
  use ExUnit.Case

  alias Annex.{
    AnnexError,
    LayerConfig
  }

  defmodule Thing do
    defstruct name: nil

    def init_layer(%LayerConfig{} = cfg) do
      case LayerConfig.fetch(cfg, :name) do
        {:ok, :name, name} ->
          %Thing{name: name}

        {:error, %AnnexError{}} = error ->
          raise error
      end
    end
  end

  test "struct defaults are correct" do
    assert %LayerConfig{} == %LayerConfig{
             module: nil,
             details: %{}
           }
  end

  describe "build/1" do
    test "works with a module" do
      assert %LayerConfig{module: Thing} = LayerConfig.build(Thing)
    end
  end

  describe "build/2" do
    test "first arg becomes the module" do
      assert %LayerConfig{module: Thing} = LayerConfig.build(Thing, [])
    end

    test "second arg can be a keyword" do
      assert %LayerConfig{
               module: Thing,
               details: %{
                 name: "Jason"
               }
             } = LayerConfig.build(Thing, name: "Jason")
    end

    test "second arg can be a map" do
      assert %LayerConfig{
               module: Thing,
               details: %{
                 name: "Jason"
               }
             } = LayerConfig.build(Thing, %{name: "Jason"})
    end

    test "raises for non-keyword|map for second arg" do
      # string not an enumerable
      assert_raise(Protocol.UndefinedError, fn ->
        LayerConfig.build(Thing, "name: Jason")
      end)

      # list is an enumerable, but not a key-value collection
      assert_raise(ArgumentError, fn ->
        LayerConfig.build(Thing, ["name: Jason"])
      end)
    end
  end

  describe "module/1" do
    test "returns the :module field" do
      cfg = %LayerConfig{
        module: Thing
      }

      assert LayerConfig.module(cfg) == Thing
    end
  end

  describe "details/1" do
    test "returns the :details field" do
      cfg = %LayerConfig{
        details: %{name: "blep"}
      }

      assert LayerConfig.details(cfg) == %{name: "blep"}
    end
  end

  describe "add/3" do
    test "adds a key and value to the LayerConfig details" do
      assert LayerConfig.add(%LayerConfig{}, :name, "Jason") == %LayerConfig{
               details: %{
                 name: "Jason"
               }
             }
    end
  end

  describe "add/2" do
    test "merges a map or keyword of details into the details of a LayerConfig" do
      cfg = %LayerConfig{details: %{name: "Jason"}}
      cfg2 = LayerConfig.add(cfg, %{name: "Jason2", other: "yep"})

      assert cfg2 == %LayerConfig{
               details: %{name: "Jason2", other: "yep"}
             }
    end
  end

  describe "init_layer/1" do
    test "Layer.t() for valid config" do
      assert Thing
             |> LayerConfig.build(name: "Jason2")
             |> LayerConfig.init_layer() == %Thing{name: "Jason2"}
    end
  end

  describe "fetch/2" do
    test "{:ok, key value} for an existing detail" do
      cfg = LayerConfig.build(Thing, name: "Jason2")
      assert {:ok, :name, "Jason2"} = LayerConfig.fetch(cfg, :name)
    end

    test "{:error, annex_error} for a non-existing detail" do
      cfg = LayerConfig.build(Thing, name: "Jason2")
      assert {:error, :not_a_key, %AnnexError{}} = LayerConfig.fetch(cfg, :not_a_key)
    end
  end

  describe "fetch_lazy/3" do
    test "returns {:ok, key, found_value} or {:ok, key, func_result}" do
      cfg = LayerConfig.build(Thing, name: "Jason2")
      {:ok, :name, "Jason2"} = LayerConfig.fetch_lazy(cfg, :name, fn -> "other" end)
      {:ok, :not_name, "other"} = LayerConfig.fetch_lazy(cfg, :not_name, fn -> "other" end)
    end
  end
end
