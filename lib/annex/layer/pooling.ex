defmodule Annex.Layer.Pooling do
  @moduledoc """
  The Pooling layer is an Annex.Layer that performs downsampling. Downsampling
  is a technique used to decrease the number of parameters in for neural
  network by combining mutliple hyperparameters into one.

  Similar to simple image compression elements that are spatially close in a
  tensor are averaged into a single element. Also similar to simple image
  compression, this operation is lossy.

  Pooling is a very useful layer (maybe absolutely necessary?) for the
  composition of a convolutional network. The typical use case for a pooling
  layer is in conjuction with a convolutional layer and an activation layer
  like so:

    data |> convolution |> pooling |> activation

  It is recommended to use the same dimensionality for pooling as the original
  data that was passed through the previous convolutional layer. For example,
  2D pooling should be used for data that was originally 2D. Data returned from
  a convolutional layer is increased in dimensionality by 1; a 2D photo becomes
  a 3D volume. A volume can be thought of as a stack of photos. Each of the
  photos in a volume has different types of features "highlighted". It may be
  effective to pool areas that are close to eachother within a single 2D layer
  of the 3D volume. However, it probably does not make much (any?) sense to
  pool between layers of the volume because each layer comes from a different
  convolution and different convolutions are "hightlighing" different features.

  For Pooling the parameters to configure are `window`, `stride`,
  `reducer`, `mapper`.

  `window` describes the spatial boundaries of the hyperparameters to be
  captured and pooled; `window` can be thought of as an outlined that surrounds
  the hyperparameters for pooling. `window` is expressed as an
  `Annex.Data.Shape.t()` which is a n-tuple of positive integers.

  `stride` describes the offset to move between captures; `stride` can be
  thought of as how far to move the box/window for each step. The stride, like
  the `window` is an `Annex.Data.Shape.t()`. However, unlike a normal shape,
  the `stride`'s does not actually describe the shape of data it describes
  how the `window` should move through the data at each step. Additionally,
  the movement described by the `stride`'s shape is not executed all at once.
  The movement describe by the `stride` is only ever execute in one dimension
  at a time. For instance, in a 2D matrix with the `stride` `{1, 2}` the
  `window` is moved through each of the windowed rows moving over 2 columns
  (2 indices) each step until the `window` reaches the end of the rows
  (2 steps goes beyond the count of columns); at the end of the rows the
  `window` will be moved down by 1 row (according to the `rows` of the 2D shape
  of our stride `{1, 2}`).

  `reducer` is a 3-arity or 2-arity function (used like `Enum.reduce/(2|3)`)
  that is used to reduce through current hyperparameters. If no mapper is
  specified, the `reducer` must return a `float`.

  `mapper` is a 1-arity function that takes any value, but must return a
  `float`. In the case that a `mapper` is not specified the result of the
  reducer is returned.

  Some examples of useful combinations for reducer and mapper are:

  ### Max Pooling

  A very common pooling technique. Starting with a reducer/2, the `reducer`
  iterates the capture keeping only the `max(acc, incoming)` eventually
  returning the maximum value of the capture. The `mapper` is omitted for
  max pooling. Max pooling is usually a better choice than average pooling
  due to it retaining the emphasis of the most influential data of the
  capture.

  ### Average Pooling

  Starting with a reducer/3 with `{0, 0.0}` as the accumulator, where the
  accumulator values are named `{count, total}` each item of the capture
  increases the `count` by 1 and is added to the `total`. The ouput of the
  `reducer`, `{count, total}` is passed to the `mapper` which divides the
  `total` by the `count` returning the `average` value of the capture.

  The process for `feedforward` can be described:
    1) create an empty pool (an empty data structure that can hold our results).
    2) go to the 0th index of each dimension.
    3) begin step n.
    4) capture the hyperparameters inside the window (in each dimension).
    5) apply the `reducer` to the captured data.
    6) apply the `mapper`, if exists, to the output of the `reducer`.
    7) put the result into the n-th place of the resulting pool.
    8) move the `window` along the data by 1 `stride`.
    9) go to 3).
  """

  defstruct window: nil,
            stride: nil,
            reducer: nil,
            mapper: nil
end
