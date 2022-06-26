# MixedTensors.jl
Implementation of general mixed tensors (i.e., tensors with any combination of covariant and contravariant indices).

## Overview
**MixedTensors** was created to facilitate computations involving vectors and linear operators in high-dimensional vector spaces, such as occur in quantum information theory.
In this package, a _tensor_ is simply a multidimensional array that represents an element of a product vector space. The space of an $(m,n)$ tensor $T$ is a product of $m$ _left spaces_ and $n$ _right spaces_; the right spaces are duals of a corresponding set of left spaces.  When tensors are multiplied together, the dimensions corresponding to matching right and left spaces are contracted, just as an matrix multiplication. Any dimenions corresponding to unmatched spaces form an outer product.

More specifically, let $V_1, V_2, V_3, \ldots$ be a collection of nominally distinct vector spaces. A tensor is an element of some product of these vector spaces and/or their duals.  $ V_{i_1} \otimes \cdots \otimes V_{i_m} \otimes V_{j_1}^\dagger \otimes \cdots \otimes V_{j_n}^\dagger$ where the $V_{i_1},\ldots,V_{i_n}$ are distinct vector spaces called the _left spaces_ and $V_{j_1},\ldots,V_{j_n}$ are distinct vectors spaces called the _right spaces_.  The _signature_ of $T$ is $(i_1,\ldots,i_m; j_1,\ldots,j_n)$.


