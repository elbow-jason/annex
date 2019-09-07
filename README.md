<p align="center">
  <a href="https://github.com/elbow-jason/annex">
    <img alt="annex logo" src="https://raw.githubusercontent.com/elbow-jason/annex/master/assets/annex_x.svg" width="450">
  </a>
</p>

<p align="center">
  A Deep Neural Network Framework for Elixir.
</p>

<p align="center">
  <a href="https://hex.pm/packages/annex">
    <img alt="Hex Version" src="https://img.shields.io/hexpm/v/annex.svg">
  </a>
  <a href="https://hexdocs.pm/annex">
    <img alt="Hex Docs" src="https://img.shields.io/badge/hex.pm-docs-green.svg?style=flat">
  </a>
</p>
<p align="center">
  <a href="https://travis-ci.com/elbow-jason/annex">
    <img alt="TravisCI Status" src="https://travis-ci.com/elbow-jason/annex.svg?branch=master">
  </a>

  <a href="https://opensource.org/licenses/MIT">
    <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-blue.svg">
  </a>
</p>

<p align="center">
  <a href="https://coveralls.io/github/elbow-jason/annex?branch=master">
    <img alt="Coveralls Test Coverage Report" src="https://coveralls.io/repos/github/elbow-jason/annex/badge.svg?branch=master">
  </a>
</p>


Annex is a framework for building and executing machine learning with deep neural networks in Elixir.

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `annex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:annex, "~> 0.2.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at [https://hexdocs.pm/annex](https://hexdocs.pm/annex).

## Features

### Layers

  - [x] Sequence
  - [x] Dense
  - [x] Activation
  - [x] Dropout
  - [ ] Convolution
  - [ ] Pooling

### Data Types (Backends)

  - [ ] List1D (list of floats)
  - [ ] List2D (list of lists of floats)
  - [ ] DMatrix (Dense Matrix) [dep](https://github.com/Qqwy/elixir-tensor)

## Extensions

### AnnexMatrex

  - 2D Annex.Data implementor that computes at native speed.
  - Uses [BLAS](http://www.netlib.org/blas/).
  - Github: [https://github.com/elbow-jason/annex_matrex/](https://github.com/elbow-jason/annex_matrex/)
  - Dependency: [https://github.com/versilov/matrex](https://github.com/versilov/matrex)


## Media

### ElixirConf 2019 Annex Presentation by Jason Goldberger

  - [YouTube](https://www.youtube.com/watch?v=Np5nSEfKLeg)
  - [Elixir Forum](https://elixirforum.com/t/elixirconf-2019-annex-introducing-an-easy-to-use-composable-deep-learning-framework-in-elixir-jason-goldberger/25189)

