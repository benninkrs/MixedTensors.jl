# AlgebraicTensors.jl
**AlgebraicTensors** implements tensors as algebraic objects (objects that can be scaled, added, and multiplied).  Building on the basic functionality of [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl), AlgebraicTensors provides a convenient way to express and performantly compute algebraic expressions involving vectors and linear operators in high-dimensonal product spaces, such as occur in quantum information science.

## Tensor Concept
In AlgebraicTensors, a tensor is a multidimensional array in which each dimension is asociated with a distinct vector space. Each vector space is identified by a label and designated as either a "left" space or as a "right" space.  Left and right spaces with the same label are dual to each other.
A tensor
 $$
 T \in V_{l_1} \otimes \cdots \otimes V_{l_m} \otimes V^\dagger_{r_1} \otimes \cdots V^\dagger_{r_n}
 $$
is represented by the type `Tensor{(l1,...,lm),(r1,...,rn)}`.  As suggested by the terminology, the array dimensions corresponding to left spaces always occur before the dimensions correpsonding to right spaces. 
The tensor element $T_{i_1,\ldots,i_m, j_1,\ldots,j_n}$ is accessed as `T[i1,...,im,j1,...,jn]`.


Left and right spaces are analogous to the columns and rows of a matrix, respectively, and the tensors implemented here may be thought of as multidimensional generalizations of vectors and matrices. In a tensor multiplication expression `A*B`, right spaces of `A` and left spaces of `B` that are dual to each other are contracted (form inner products).


## Tensor Construction
The standard way to construct a tensor is by providing an array and the spaces associated with its dimensions:
```
julia> using AlgebraicTensors

julia> a = randn(2,3,4);

julia> t = Tensor{(14,10),(5,)}(a)

julia> size(t)			# array size
(2,3,4)

julia> nlspaces(t)		# number of left spaces
2

julia> lspaces(t)		# left spaces
(14, 10)

julia> lsize(t)			# size of the "left" dimensions of array
(2, 3)

julia> nrspaces(t)		# number of right spaces
1

julia> rspaces(t)		# right spaces
(5,)

julia> spaces(t)		# left and right spaces concatenated
(14, 10, 5)

julia> t[2,1,3]			# a[2,1,3]
```
A multidimensional vector is created by specifying an empty tuple for the right spaces:
```
julia> v = Tensor{(6,9),()}([1 2 3; 4 5 6])		# construct a vector
```
Once constructed a tensor's spaces cannot be changed, but a convenient syntax enables one to create a new tensor with the same backing array and different spaces:
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

julia> S^3					# powers (square tensors only)

julia> exp(S)				# operator exponentiation (square tensors only)

julia> tr(X, 3)				# (partial) trace

julia> transpose(X, 5)		# (partial) transpose

julia> eig(S)					# eigenvalues (square tensors only)
```
Some tensor operations are defined only for certain combinations of left and right spaces.  For example, addition and subtraction are defined only for tensors with the same spaces.  Likewise, analytic functions are defined only for "square" tensors: tensors whose left and right spaces are the same, and whose corresponding left and right array dimensions have the same axes.


## Other Functions

A `Tensor` can be converted to a `Matrix` by folding all the left spaces into the first dimension and all the right spaces into the second dimension:
```
julia> Matrix(t)
```
This is sometimes helpful for inspecting tensors that represent linear operators.


## Implementation

The `Tensor` type is a wrapper for a multidimensional array, with type parameters specifying the associated left and right spaces.  This enables the generation of performant code since the dimensions to be contracted (and hence the requisite looping constructs) are often determinable at compile-time.  However, if most of the tensor operations involve tensors whose spaces are run-time values, this benefit may not be fully realized.  Also, if one's calculation consists of many different contractions involving many different sets of spaces, compile time may become non-negligible.

Most of the tensor operations provided by AlgebraicTensors are efficiently implemented via the low-level functions provided by [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl).



## Comparison with Other Tensor Packages

AlgebraicTensors complements existing Julia tensor packages:
 * In [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl), tensors can have only 1, 2, or 4 dimensions and maximum length 3 in any dimension.  AlgebraicTensors supports tensors of (practically) arbitrary size and dimensionality.
 * [Einsum.jl](https://github.com/ahwillia/Einsum.jl) and the macros provided by [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) provide an index-based syntax similar to Einstein notation for implementing contraction and other operations on multidimensional arrays.  This requires a labelling of the dimensions, which determine the contraction pattern, to be "hard coded" into the expression. AlgebraicTensors implements tensors as algebraic objects that can be multiplied and added using standard syntax, with the dimensions to be contracted (or not) determined programmatically from the spaces associated with each tensor. 
 * [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) is the most similar package. It too implements tensors as multilinear maps, however with an emphasis on category theoretic aspects.
     * In **TensorKit**, vector spaces are first-class objects defined by their dimensionality and underlying scalar field.  However, vector spaces that are mathematically equivalent are not distinguished. Tensor multiplication `A*B` either requires the domain of `A`` to coincide with the codomain of `B` or requires an explicit indication of dimension pairs to be contracted.
	  * In **AlgebraicTensors**, vector spaces are never explicitly defined; they are simply designated by arbitrary labels. The labels distinguish vector spaces and give them identity apart from any mathematical properties that might be imputed to them. In tensor multiplication `A*B`, the vector space labels determine whether the operation is valid, which dimensions of `A` and `B` should be contracted, and which dimensions should form outer products.   