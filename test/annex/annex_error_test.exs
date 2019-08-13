defmodule Annex.AnnexErrorTest do
  use ExUnit.Case
  alias Annex.AnnexError

  describe "exception" do
    test "can be raised" do
      err = %AnnexError{
        message: "Hello",
        details: [to: "World"]
      }

      assert_raise(AnnexError, fn -> raise err end)
    end
  end

  describe "build/2" do
    test "returns a built AnnexError struct" do
      assert AnnexError.build("Hello", to: "World") == %AnnexError{
               message: "Hello",
               details: [to: "World"]
             }
    end
  end

  describe "add_details/2" do
    test "appends keywords to details" do
      err1 = AnnexError.build("Hello", to: "World")
      err2 = AnnexError.add_details(err1, also: "Mom")

      assert err2 == %AnnexError{
               message: "Hello",
               details: [to: "World", also: "Mom"]
             }
    end
  end

  describe "message/1" do
    test "returns a rendered string of the AnnexError" do
      assert "Hello"
             |> AnnexError.build(to: "World")
             |> AnnexError.message() == """
             Hello
             to: "World"
             """
    end

    test "renders sub-keywords nested" do
      assert "Hello"
             |> AnnexError.build(to: [name: "World"])
             |> AnnexError.message() == """
             Hello
             to: [
               name: "World"
             ]
             """
    end

    test "renders :code keyword as an unwrapped/non-inspected string" do
      code = "1 + 1 == 2"

      assert "Hello"
             |> AnnexError.build(code: code, other: "THING")
             |> AnnexError.message() == """
             Hello
             code: 1 + 1 == 2
             other: "THING"
             """
    end
  end
end
