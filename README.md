# MixedTensors.jl
Implementation of tensors of any size and indexed by any combination of covariant and contravariant indices.

## Overview
**MixedTensors** is a package to facilitate computations involving vectors and linear operators in high-dimensonal product vector spaces, such as occur in quantum information theory.  

For this purpose a **tensor** is a multidimensional array, each dimension of which is asociated with particular vector space.  Let $V_1, V_2, V_3, \ldots$ be a set of distinguishable vectors spaces, and let $V^\dagger_i$ denote the dual of $V_i$.  A tensor $T$ with _left spaces_ $(l_1,..,l_m)$ and _right spaces_ $(r_1,\ldots,r_n)$ represents an element of $V_{l_1} \otimes \cdots \otimes V_{l_m} \otimes V^\dagger_{r_1} \otimes \cdots V^\dagger_{r_n}$.  Tensors may regarded as multdimensional generalizations of vectors and matrices:  When two tensors are multiplied $AB$, the right dimensions of $A$ are contracted with the matching left dimensions of $B$; the remaining dimensions form an outer product. 

This package provides the `Tensor` type to represent a tensor in the sense just described, as well as implementations for many mathematical operations.

## Usage

### Constructing Tensors
The typical way to construct a tensor is by providing an array and the spaces associated with its dimensions:
```
julia> t = Tensor(randn(2,3,4), (14,10), (5))
```
creates a random tensor of size (2,3,4) that exists in the space $V_{14} \otimes V_{10} \otimes V^\dagger_5$. The first dimension of `t` is associated with $V_{14}$, the second dimension is associated with $V_{10}$, and the third is associated with $V^\dagger_{5}$.  Note that the spaces need not be in any particular order.  However, all the left spaces must be distinct, as must all the right spaces. Furthermore, the total number of spaces must equal the number of dimensions of the array.

A multidimensional (left) vector is created by specifying an empty tuple for the right spaces:
```
julia> v = Tensor([1 2 3; 4 5 6]; (6,9), ())
```
Similarly, a multidimensional right (dual) vector is created by specificying an empty tuple for the left spaces.  If a single set of spaces are given, they are used for both the left and right spaces:
```
julia> s = Tensor(randn(2,3,2,3), (7,4))   # left spaces = right spaces = (7,4)
```
For convenience, the spaces associated with a tensor can be changed using a function-like syntax:
```
t((5,6,8),(1,4))      # keep the data, change the spaces
```
yields a new tensor that has the same data as `t` but has left spaces (5,6,8) and right spaces (1,4).

The left and right spaces of a tensor, and the number of left and right spaces, can be queried thusly:
```
julia> lspaces(t)
(14, 10)

julia> nlspaces(t)
2

julia> rspaces(t)
(5,)

julia> nrspaces(t)
1

julia> spaces(t)
((14,10), (5,))
```
Note that `spaces` shows the spaces associated with each dimensions, in order.  It is important to note that the same tensor can be represented in multiple ways, by permuting the dimensions of the array and permuted the associated spaces accordingly.  Thus
```
julia> A = Tensor([1 2; 3 4], (5,7), ());
julia> B = Tensor(transpose([1 2; 3 4]), (7,5), ())
julia> A == B
true
```
### Indexing




