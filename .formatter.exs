# Used by "mix format"

locals_without_parens = [
  debug_assert: 2
]

# final step is returning the formatter configuration options
[
  inputs: ["{mix,.formatter}.exs", "{config,lib,test}/**/*.{ex,exs}"],
  locals_without_parens: locals_without_parens
]
