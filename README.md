# AlgebraicTensors.jl
An implementation of tensors as algebraic objects:  objects that can be scaled, added, and multiplied.  This facilitates computations involving vectors and linear operators in high-dimensonal product spaces, such as occur in quantum information theory.

## Overview
As implemented here, a tensor is a multidimensional array where each dimension is asociated with a particular vector space.  Each of these spaces is designated as either a "left" space or as a "right" space; left and right spaces with the same label are dual to each other. In multiplication expressions, a tensor's left spaces contract (form inner products) with dual factors on the left, whereas right spaces form inner products with dual factors on the right. Left and right spaces are analogous to the columns and rows of a matrix, respectively, and the tensors implemented here may be thought of as multidimensional generalizations of vectors and matrices.

A tensor is represented by the provided `Tensor` type. A `Tensor` `T` with left spaces `(l1,...,lm)` and right spaces `(r1,...,rn)` represents a tensor
 $$
 T \in V_{l_1} \otimes \cdots \otimes V_{l_m} \otimes V^\dagger_{r_1} \otimes \cdots V^\dagger_{r_n}
 $$
where $V_1, V_2, V_3, \ldots$ are an implicit global set of vector spaces. The tensor elements $T_{i_1,\ldots,i_m, j_1,\ldots,j_n}$ are accessed as `T[i1,...,im,j1,...,jn]`.  Many mathematical operations involving tensors are implemented also.  Below is a quick introduction to the functionality provided by this package; the complete functionality is described in the documentation.  


## Tensor Construction
The standard way to construct a tensor is by providing an array and the spaces associated with its dimensions:
```
julia> using AlgebraicTensors

julia> a = reshape(1:24, (2,3,4));

julia> t = Tensor(a, (14,10), (5))

julia> size(t)			# array size
(2,3,4)

julia> nlspaces(t)		# number of left spaces
2

julia> lspaces(t)		# left spaces
(14, 10)

julia> lsize(t)			# size of the "left" dimensions of array
(2,3)

julia> spaces(t)		# left and right spaces
((14,10), (5,))

julia> t[2,1,3]			# a[2,1,3]
```
A multidimensional vector is created by specifying an empty tuple for the right spaces:
```
julia> v = Tensor([1 2 3; 4 5 6], (6,9), ())		# construct a vector
```
Once constructed a tensor's spaces cannot be changed, but a convenient syntax enables one to create an alias of the tensor with different spaces:
```
julia> t((5,6,8),(1,4))      # keep the data, set the spaces to (5,6,8),(1,4)
```
Note that due to the freedom of operator ordering, the same mathematical tensor can be represented via different `Tensor` objects whose spaces and underlying array dimensions are in different orders. While most operations are insensitive to such mathematically equivalent orderings, operations that directly deal with array elements or dimensions (such as indexing) obviously depend on the chosen ordering. 

## Some Tensor Operations
```
julia> 2*X					# scaling

julia> A + B				# addition/subtraction

julia> C * D				# multiplication

julia> A == B				# equality (yields a Bool)

julia> X'					# Hermitian adjoint

julia> S^3					# powers

julia> exp(S)				# operator exponentiation

julia> tr(X, 3)				# (partial) trace

julia> transpose(X, 5)		# (partial) transpose

julia> eig(S)					# eigenvalues

```
Some tensor operations are defined only for certain combinations of left and right spaces.  For example, addition and subtraction are defined only for tensors with the same spaces.  Likewise, analytic functions are defined only for "square" tensors: tensors whose left and right spaces are the same, and whose corresponding left and right array dimensions have the same axes.


## Miscellaneous

A `Tensor` can be converted to a `Matrix` by folding all the left spaces into the first dimension and all the right spaces into the second dimension:
```
julia> Matrix(t)
??
```
This is sometimes useful for inspecting tensors that represent linear operators.