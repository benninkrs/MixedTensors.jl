"""
module MultiMatrices

Multimatrices generalize matrices to tensor product spaces. A `MultiMatrix` of
size (l1,…,ln,r1,…,rn) represents a linear map between a "left"
vector space L of size (l1,...,ln) and a "right" vector space R of size
(r1,...,rn). (In the language of tensors, a multimatrix is a tensor with an equal
number of "up" and "down" indices.) `n` is called the _number of spaces_, while
`2n` is the number of _dimensions_.
"""


module MultiMatrices

export MultiMatrix, lsize, rsize, nspaces

using BaseExtensions
using StaticArrays
using TensorOperations: trace!
using TypeTools: asdeclared

import Base: ndims, length, size, axes
import Base: reshape, permutedims, adjoint, transpose, Matrix
import Base: getindex, setindex!
#import Base: display, summary, array_summary
import Base.(*)
import Base: exp, log, sin, cos, tan, sinh, cosh, tanh

import LinearAlgebra: tr, eigvals, svdvals



"""
`MultiMatrix(A::AbstractArray)` creates a multimatrix from `A`. `ndims(A)` must be even.
"""
struct MultiMatrix{T, N, A<:AbstractArray} <: AbstractArray{T,N}
	data::A
	function MultiMatrix(d::A) where A<:AbstractArray{T,N} where {T,N}
		if !iseven(N)
			error("Input array must have an even number of dimensions")
		else
			return new{T,N,A}(d)
		end
	end
end


# Size and shape
ndims(m::MultiMatrix) = ndims(m.data)
nspaces(mm::MultiMatrix) = ndims(mm) >> 1
length(m::MultiMatrix) = length(m.data)
size(m::MultiMatrix) = size(m.data)
size(m::MultiMatrix, dims) = size(m.data, dims)
lsize(m::MultiMatrix) =  ntuple(d -> size(m.data, d), nspaces(m))
rsize(m::MultiMatrix) = ntuple(d -> size(m.data, d+nspaces(m)), nspaces(m))
axes(m::MultiMatrix) = axes(m.data)
axes(m::MultiMatrix, d) = axes(m.data, d)


# Conversions
convert(t, mm::MultiMatrix) = convert(t, mm.data)

"""
Return a `Matrix` obtained by reshaping a `MultiMatrix`. The first `n`
dimensions are combined into the first dimension of the matrix, while the last
`n` dimensions are combined into the second dimension of the matrix.
"""
Matrix(m::MultiMatrix) = reshape(m.data, prod(lsize(m)), prod(rsize(m)) )


# Array access
# If accessing a single element, return that element.
# Otherwise (i.e. if requesting a range) return a MultiMatrix.
getindex(m::MultiMatrix, i::Vararg{Integer}) = getindex(m.data, i...)
getindex(m::MultiMatrix, i::CartesianIndex) = getindex(m.data, i)
getindex(m::MultiMatrix, i...) = MultiMatrix(getindex(m.data, i...))

setindex!(m::MultiMatrix, i...) = setindex!(m.data, i...)



reshape(m::MultiMatrix, shape) = MultiMatrix(reshape(m.data, shape))

permutedims(m::MultiMatrix, ord) = MultiMatrix(permutedims(m.data, ord))

# Permute spaces (pairs of dimensions).
function permute_spaces(m::MultiMatrix, p)
	return MultiMatrix(permutedims(m.data), [p; p+nspaces(m)])
end


#Swap left and right dimensions of a MultiMatrix
function transpose(mm::MultiMatrix)
	n = nspaces(mm)
	return MultiMatrix(permutedims(mm.data, [n+1:2*n; 1:n]))
end

# Partial transpose
function transpose(mm::MultiMatrix, spaces)
	n = nspaces(mm)
	lorder = collect(1:n)
	lorder[spaces] += n
	rorder = collect(n+1:2*n)
	rorder[spaces] -= n
	return MultiMatrix(permutedims(m.data, [lorder; rorder]))
end


# TODO: Make a lazy wrapper, just like Base does.
# Note, "partial adjoint" doesn't really make sense.
function adjoint(mm::MultiMatrix)
	n = nspaces(mm)
	# adjoint is called element-by-element (i.e. it recurses as required)
	return MultiMatrix(permutedims(adjoint.(mm.data), [n+1:2*n; 1:n]))
end



"""
`tr(A)` returns the trace of a `MultiMatrix` `A`, i.e. it contracts each left
dimension with the corresponding right dimension and returns a scalar.

`tr(A, spaces)` traces out the indicated spaces, returning another `MultiMatrix`
(even if all the spaces are traced).
"""
function tr(A::MultiMatrix)
	lsize(A) == rsize(A) || error("Matrix must be square (lsize(A) == rsize(A))")
	return tr(Matrix(A))
end

tr(A::MultiMatrix, i::Integer) = tr(A, (i,))

function tr(arr::MultiMatrix, tspaces::Dims)
	n = nspaces(arr)
	lsz = lsize(arr)
	rsz = rsize(arr)
	lsz[tspaces] == rsz[tspaces] || error("Traced dimensions must be equal (lsize(A)[] == rsize(A))")


	mask = trues(n)
	for t in tspaces
		if t<1 || t>nspaces(arr)
			error("Invalid spaces to be traced")
		else
			mask[t] = false
		end
	end
	kspaces = oneto(n)[mask]		# findall(mask) returns a vector, not a tuple

	return tr_dims(arr, tspaces, kspaces)
end


function tr_dims(arr::MultiMatrix{T,N,A}, tdims, kdims::Dims{K}) where {T,N,A,K}
	nspc = nspaces(arr)
	lsz = lsize(arr)
	rsz = rsize(arr)
	R = asdeclared(A){T,2*K}(undef, lsz[kdims]..., rsz[kdims]...)
	trace!(1, arr.data, :N, 0, R, kdims, kdims+nspc, tdims, tdims+nspc)
	return MultiMatrix(R)
end



# Matrix multiplication.  These methods are actually quite fast -- about as fast as the
# core matrix multiplication. We appear to incur very little overhead.

"""
For multimatrices `A` and `B`, A*B returns the result of contracting the right
dimensions of `A` with the `left` dimensions of `B`.

If `B` is an `AbstractArray`, the first `nspaces(A)` of `B` are contracted with
the right dimensions of `A`.  Similarly, if `A` is an `AbstractArray`, the last
`nspaces(B)` dimensions of `A` are contracted withe the left dimensions of `B`.
"""
function (*)(A::MultiMatrix, B::MultiMatrix)
	lszA = lsize(A)
	rszA = rsize(A)
	lszB = lsize(B)
	rszB = rsize(B)
	rszA == lszB || error("rsize(A) must equal lsize(B)")
	return MultiMatrix(reshape(Matrix(A) * Matrix(B), (lszA...,rszB...)))
end


function (*)(A::MultiMatrix, B::AbstractArray)
	n = nspaces(A)
	lszA = lsize(A)
	rszA = rsize(A)
	lszB = size(B, 1:n)
	rszB = size(B, n+1:ndims(B))
	rszA == lszB || error("rsize(A) must equal size(B, 1:nspaces(A))")
	MA = Matrix(A)
	MB = reshape(B, prod(lszB), prod(rszB))
	return reshape(MA * MB, (lszA...,rszB...))
end


function (*)(A::AbstractArray, B::MultiMatrix)
	n = nspaces(B)
	lszB = lsize(B)
	rszB = rsize(B)
	ndims(A) >= n || error("A must have at least nspaces(B) dimensions")
	lszA = size(B, 1:ndims(A)-n)
	rszA = size(A, ndims(A)-n+1:ndims(A))
	rszA == lszB || error("lsize(B) must equal the size of the last nspaces(B) dimensions of A")
	MA = reshape(A, prod(lszA), prod(rszA))
	MB = Matrix(B)
	return reshape(MA * MB, (lszA...,rszB...))
end


# Analytic matrix functions

for f in [:exp, :log, :sin, :cos, :tan, :sinh, :cosh, :tanh]
	@eval $f(m::MultiMatrix) = MultiMatrix($f(m.data))
end


# Other linear algebra stuff

eigvals(A::MultiMatrix, args...) = eigvals(Matrix(A), args...)
svdvals(A::MultiMatrix, args...) = svdvals(Matrix(A), args...)



#
# """
# `*(A::MultiMatrix, spaces...) * B::MultiMatrix` contracts A with the specified subspaces of B,
# where `ndims(A) <= ndims(B)`, `length(spaces) = ndims(A)`, and `spaces ⊆ 1:ndims(B)`.
# This is equivalent to, but generally faster than, tensoring A with the identity operator, permuting, and multiplying with B.
#
# `A * (B, spaces)` contracts B along the specified subspaces of A.
# """
# function (*)(tup::Tuple{MultiMatrix, Vararg{Int}}, B::MultiMatrix)
# 	A = tup[1];
# 	dims = collect(tup[2:end])
#
# 	a_lsz = lsize(A)
# 	a_rsz = rsize(A)
# 	b_lsz = lsize(B)
# 	b_rsz = rsize(B)
#
# 	length(dims) == ndims(A) ||
# 		error("Incompatible arguments: length(spaces) == $(length(dims)), should equal ndims(A) = $(ndims(A))")
#
# 	a_rsz == b_lsz[dims] ||
# 		error("Dimension mismatch: rsize(A) == $a_rsz, lsize(B)[dims] = $(b_lsz[dims])")
#
# 	# make a mask of which spaces of B are involved in the multiply
# 	other_dims = compl_dims(dims, ndims(B));
#
# 	# reorder the dimensions of B for multiplication and convert to a matrix
# 	order = [dims; other_dims; (ndims(B)+1:2*ndims(B))...]
# 	b_lsz_mult = *(b_lsz[dims]...);
# 	b_rsz_mult = *(b_lsz[other_dims]..., b_rsz...)
# 	B_mat = reshape(permutedims(B.data, order), (b_lsz_mult, b_rsz_mult))
#
# 	# Compute the matrix product, reshape, and put the dimensions back in original order
# 	C_mat = matrix(A) * B_mat
# 	c_lsz = (a_lsz..., b_lsz[other_dims]...)
# 	C_mat = ipermutedims(reshape(C_mat, (c_lsz..., b_rsz...)), order)
# 	return MultiMatrix(C_mat)
# end
#
# function mult(tup::Tuple{MultiMatrix, Vararg{Int}}, B::MultiMatrix)
# 	A = tup[1]
# 	ndA = ndims(A)
# 	ndB = ndims(B)
# 	dims = collect(tup[2:end])
# 	other_dims = compl_dims(dims, ndB)
#
# 	a_lsz = lsize(A)
# 	a_rsz = rsize(A)
# 	b_lsz = lsize(B)
# 	b_rsz = rsize(B)
#
# 	order = [dims; other_dims; (ndims(B)+1:2*ndims(B))...]
# 	#iorder = invperm(order)
# 	c_sz = Array{Int}(2*ndB);
# 	c_sz[order] = [a_lsz...; b_lsz[other_dims]...; b_rsz...]
#
# 	new_data = Array{promote_type(eltype(A),eltype(B))}(c_sz...)
# 	oA = 1:ndA
# 	cA = ndA+1:2*ndA
# 	oB = [other_dims...; ndB+1:2*ndB]
# 	cB = dims
# 	iC = Array{Int}(2*ndB);
# 	iC[order] = 1:2*ndB
# 	contract!(1, A.data, Val{:N}, B.data, Val{:N}, 0, new_data, oA, cA, oB, cB, iC, Val{:BLAS})
# 	return MultiMatrix(new_data)
# end
#
#
# function (*)(A::MultiMatrix, tup::Tuple{MultiMatrix, TTuple{Int}})
# 	B = tup[1];
# 	dims = collect(tup)[2:end]
# 	a_lsz = lsize(A)
# 	a_rsz = rsize(A)
# 	b_lsz = lsize(B)
# 	b_rsz = rsize(B)
#
# 	length(dims) == ndims(B) ||
# 		error("Incompatible arguments: length(spaces) == $(length(dims)), should equal ndims(B) = $(ndims(A))")
#
# 	b_lsz == a_rsz[dims] ||
# 		error("Dimension mismatch: lsize(B) == $b_lsz, rsize(A)[dims] = $(b_lsz[dims])")
#
# 	# make a mask of which spaces of A are involved in the multiply
# 	other_dims = compl_dims(dims, ndims(A))
# 	nd = dims(A);
#
# 	# reorder the dimensions of A for multiplication and convert to a matrix
# 	order = [(1:nd)...; nd+other_dims; nd+dims]
# 	a_lsz_mult = *(a_lsz..., a_rsz[other_dims]...)
# 	a_rsz_mult = *(a_rsz[dims]...)
# 	A_mat = reshape(permutedims(A.data, order), (a_lsz_mult, a_rsz_mult))
#
# 	# Compute the matrix product, reshape, and put the dimensions back in original order
# 	C_mat = A_mat * matrix(B)
# 	c_rsz = (a_rsz[other_dims]..., b_rsz...);
# 	C_mat = ipermutedims(reshape(C_mat, (a_lsz..., c_rsz...)), order);
# 	return MultiMatrix(C_mat)
# end
#
#
# trace(X::MultiMatrix) = trace(matrix(X))
# # Partial trace
# trace(X::MultiMatrix, dims...) = trace(X, collect(dims))
# trace(X::MultiMatrix, dims::Array{Int,1}) = trace_(X, dims, compl_dims(dims, ndims(X)))
#
# function trace_(X::MultiMatrix, dims::Array{Int,1}, other_dims::Array{Int,1})
# 	if isempty(dims)
# 		return X
# 	elseif isempty(other_dims)
# 		return trace(matrix(X))
# 	end
#
# 	nd = ndims(X);
# 	lsz = lsize(X);
# 	rsz = rsize(X);
#
# 	order = [other_dims; nd+other_dims; dims; nd+dims];
# 	new_data = permutedims(X.data, order);
# 	tsz = (lsz[dims]..., rsz[dims]...);
# 	xsz = (lsize(X)[other_dims]..., rsize(X)[other_dims]...)
# 	new_data = reshape(new_data, *(xsz...), *(tsz...));
# 	# WRONG!  This needs to be done for each dimension
# 	new_data = sum(new_data[:, 1:tsz[1]+1:tsz[1]*tsz[2]], 2);
# 	new_data = reshape(new_data, xsz);
# 	return MultiMatrix(new_data);
# end
#
#
# # Given an array representing a subset of 1:n, returns an array of the complentary subset.
# function compl_dims(dims::Array{Int}, n::Int)
# 	is_active = falses(n)
# 	is_active[dims] = true;
# 	return find(!is_active);
# end
#
#
# # Reduce (trace over complementary dimensions)
# reduce(X::MultiMatrix, dims...) = reduce(X, collect(dims))
# reduce(X::MultiMatrix, dims::Array{Int,1}) = trace_(X, compl_dims(collect(dims), ndims(X)), dims)




end
