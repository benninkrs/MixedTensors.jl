"""
	MultiMatrices (module)

Multimatrices generalize matrices to tensor product spaces. A `MultiMatrix` of
size (l1,…,ln,r1,…,rn) represents a linear map between a "left"
vector space L of size (l1,...,ln) and a "right" vector space R of size
(r1,...,rn). (In the language of tensors, a multimatrix is a tensor with an equal
number of "up" and "down" indices.)
"""

#=
TODO: Use ldims/rdims or lperm/rperm?
TODO: Use dispatch to separate out A*B with same spaces
TODO: Extend addition, subtraction to allow spaces to be in different order
TODO: Extend * to form lazy outer products
TODO: Support +,- for dissimilar spaces?  Perhaps lazy structures?
TODO: Figure out why broadcasting is slow
TODO: Use Strided.jl?
TODO: Support in-place operations?
TODO: Check validity of dims in lsize, rsize, laxes, raxes
TODO: generalize + to >2 arguments
TODO: Better support of different underlying array types
			In places (e.g. *) it is assumed that the underlying array type has
			the element type as the first parameter and has a constructor of the
			form ArrayType{ElementType}(undef, size).
		* OR *
			Back with Array type only?  No, this limits future exensibility. Also, it would
			make it harder to construct from a non-standard array.
=#

module MultiMatrices

export MultiMatrix, multivector, lsize, rsize, lspaces, rspaces, nlspaces, nrspaces
#export ldim, rdim

using MiscUtils
using SuperTuples
using StaticArrays
using LinearAlgebra
using TensorOperations: trace!, contract!
using Base.Broadcast: Broadcasted, BroadcastStyle
using PermutedIteration

using Base: promote_op

import Base: show
import Base: ndims, length, size, axes, similar
import Base: reshape, permutedims, adjoint, transpose, Matrix, ==
import Base: getindex, setindex!
#import Base: display, summary, array_summary
import Base: (+), (-), (*), (/), (^)
import Base: inv, exp, log, sin, cos, tan, sinh, cosh, tanh

import LinearAlgebra: tr, eigvals, svdvals, opnorm
import Base: BroadcastStyle, similar



#--------------------------------------


const Iterable = Union{Tuple, AbstractArray, UnitRange, Base.Generator}

# Alias for whatever integer type we use to specify a set of vector spaces
# Not to be confused with DataStructures.IntSet or Base.BitSet.
const IntSet = UInt128

"""
	MultiMatrix{LS, RS, T, N, A<:AbstractArray{T,N}}	(struct)

A `MultiMatrix` represents a linear map between vector spaces on its left and right,
each of which is generally a tensor product space.
`LS` specifies the vector spaces acted upon on the left; `RS` specifies the vector
spaces acted upon on the right.
"""

struct MultiMatrix{LS, RS, T, N, A<:AbstractArray{T,N}, NL, NR} <: AbstractArray{T,N}
	data::A
	lperm::Dims{NL}		# order of left spaces (i'th space in LS <==> axes(data, lperm[i])
	rperm::Dims{NR}		# order of right spaces (i'th space in RS <==> axes(data, NL + rperm[i])

	# Constructor with input validation
	function MultiMatrix{LS, RS, T, N, A}(data::A, lperm::NTuple{NL,Integer}, rperm::NTuple{NR,Integer}) where {LS, RS, NL, NR} where {A<:AbstractArray{T,N}} where {T,N}
		count_ones(LS) == NL || error("length(lperm) must equal count_ones(LS)")
		count_ones(RS) == NR || error("length(rperm) must equal count_ones(RS)")
		NL + NR == N || error("ndims(A) must be equal the total number of spaces (left + right)")
		return new{LS, RS, T, N, A, NL, NR}(data, lperm, rperm)
	end
end

# Convenience constructors
"""
	MultiMatrix(A::AbstractArray)
	MultiMatrix(A::AbstractArray, spaces::Tuple)
	MultiMatrix(A::AbstractArray, lspaces::Tuple, rspaces::Tuple)

Create from array 'A' a MultiMatrix that acts on specified vector spaces.
The elements of `spaces` (or `lspaces` or `rspaces`) must be distinct.
If omitted, `spaces` is taken to be `(1,...,ndims(A)/2)`.
`rspaces` defaults to `lspaces`.
`length(lspaces) + length(rspaces)` must equal `ndims(A)`.

 To construct a MultiMatrix with default left spaces and no right spaces, use
 [`multivector(A)`](@ref).
"""
function MultiMatrix end

# Construct from array with default spaces
function MultiMatrix(arr::A) where A<:AbstractArray{T,N} where {T,N}
	iseven(N) || error("Source array must have an even number of dimensions")
	M = N >> 1
	LS = IntSet(N-1)		# trick to create 001⋯1 with M ones
	MultiMatrix{LS,LS,T,N,A}(arr, oneto(M), oneto(M))
end

# Construct from array with custom spaces.
MultiMatrix(arr, spaces::Dims) = MultiMatrix(arr, Val{spaces})
MultiMatrix(arr, lspaces::Dims, rspaces::Dims) = MultiMatrix(arr, Val{lspaces}, Val{rspaces})

# Would it make sense to have a constructor that takes LS,RS parameters explicitly?

# Construct from array with custom spaces - type inferred.
function MultiMatrix(arr::A, ::Type{Val{S}}) where {S} where A<:AbstractArray{T,N} where {T,N}
	perm = sortperm(S)
	SI = binteger(IntSet, Val{S})
	MultiMatrix{SI,SI,T,N,A}(arr, perm, perm)
end

function MultiMatrix(arr::A, ::Type{Val{LS}}, ::Type{Val{RS}}) where {LS, RS} where A<:AbstractArray{T,N} where {T,N}
	lperm = sortperm(LS)
	rperm = sortperm(RS)
	LSI = binteger(IntSet, Val{LS})
	RSI = binteger(IntSet, Val{RS})
	MultiMatrix{LSI,RSI,T,N,A}(arr, lperm, rperm)
end


multivector(arr, S) = MultiMatrix(arr, Val{S}, Val{()})
multivector(arr, ::Type{Val{S}}) where {S} = MultiMatrix(arr, Val{S}, Val{()})


# Reconstruct with different spaces
"""
	(M::MultiMatrix)(spaces...)
	(M::MultiMatrix)(spaces::Tuple)
	(M::MultiMatrix)(lspaces::Tuple, rspaces::Tuple)`

Create a MultiMatrix with the same data as `M` but acting on different spaces.
(This is a lazy operation that is generally to be preferred over [`permutedims`](@ref).)
"""
(M::MultiMatrix)(spaces::Vararg{Int64}) = M(spaces)

(M::MultiMatrix)(spaces::Dims) = MultiMatrix(M.data, Val{spaces})
(M::MultiMatrix)(lspaces::Dims, rspaces::Dims) where {N} = MultiMatrix(M.data, Val{lspaces}, Val{rspaces})
(M::MultiMatrix)(::Type{Val{S}}) where {S} = MultiMatrix(M.data, Val{S})
(M::MultiMatrix)(::Type{Val{LS}}, ::Type{Val{RS}}) where {LS,RS} = MultiMatrix(M.data, Val{LS}, Val{RS})


# Similar array with the same spaces and same size

# similar(M::MultiMatrix{LS,RS,T,N,A}) where {LS,RS,T,N,A} = MultiMatrix{LS,RS,T,N,A}(similar(M.data))
# # Similar array with different type, but same size and spaces.
# similar(M::MultiMatrix{LS,RS,T,N}, newT) = MultiMatrix{LS,RS,newT,N}(similar(M.data, newT), M.lperm, M.rperm)
#
# #Similar array with different type and/or size
# const Shape = Tuple{Union{Integer, Base.OneTo},Vararg{Union{Integer, Base.OneTo}}}
#
# similar(::Type{M}) where {M<:MultiMatrix{LS,RS,T,N,A}} where {LS,RS} where {A<:AbstractArray{T,N}} where {T,N} = MultiMatrix{LS,RS,T,N,A}(similar(A, shape))

#-------------
# Size and shape
ndims(M::MultiMatrix) = ndims(M.data)

# Find the dimension of M.data corresponding to the i'th left or right space
# (Internal use only)
ldim(M::MultiMatrix, i) = M.lperm[i]
rdim(M::MultiMatrix, i) = length(M.lperm) + M.rperm[i]

lspaces(M::MultiMatrix{LS}) where {LS} = invpermute(findbits(LS), M.lperm)
rspaces(M::MultiMatrix{LS, RS}) where {LS,RS} = invpermute(findbits(RS), M.rperm)
nlspaces(M::MultiMatrix{LS, RS, T, N, A, NL, NR}) where {LS, RS, T, N, A, NL, NR} = NL
nrspaces(M::MultiMatrix{LS, RS, T, N, A, NL, NR}) where {LS, RS, T, N, A, NL, NR} = NR

length(M::MultiMatrix) = length(M.data)

size(M::MultiMatrix) = size(M.data)
size(M::MultiMatrix, dim) = size(M.data, dim)
size(M::MultiMatrix, dims::Iterable) = size(M.data, dims)

lsize(M::MultiMatrix) = ntuple(d -> size(M.data, d), nlspaces(M))
lsize(M::MultiMatrix, dim) =  size(M.data, dim)
lsize(M::MultiMatrix, dim::Iterable) =  map(d -> size(M.data, d), dims)

rsize(M::MultiMatrix) = ntuple(d -> size(M.data, d+nlspaces(M)), nrspaces(M))
rsize(M::MultiMatrix, dim) =  size(M.data, dim + nlspaces(M))
rsize(M::MultiMatrix, dim::Iterable) =  map(d -> size(M.data, d + nspaces(M)), dims)

axes(M::MultiMatrix) = axes(M.data)
axes(M::MultiMatrix, dim) = axes(M.data, dim)
axes(M::MultiMatrix, dims::Iterable) = map(d->axes(M.data, d), dims)


"""
`laxes(M)` left axes of `M`.
"""
#laxes(M::MultiMatrix) = ntuple(d -> axes(M.data, d), nspaces(M))		# inexplicably, this doesn't infer even though raxes(M) does
laxes(M::MultiMatrix) = map(d -> axes(M.data, d), oneto(nlspaces(M)))
laxes(M::MultiMatrix, dim) = axes(M.data, dim)
laxes(M::MultiMatrix, dims::Iterable) = map(d -> axes(M.data, d), dims)

"""
`raxes(M)` right axes of `M`.
"""
raxes(M::MultiMatrix) = ntuple(d -> axes(M.data, d+nlspaces(M)), nrspaces(M))
raxes(M::MultiMatrix, dim) = axes(M.data, dim + nlspaces(M))
raxes(M::MultiMatrix, dims::Iterable) = map(d -> axes(M.data, d + nspaces(M)), dims)


arraytype(::MultiMatrix{LS,RS,T,N,A} where {LS,RS,T,N}) where A = A

# # Return the type of an array
# function promote_arraytype(A::MultiMatrix, B::MultiMatrix)
# 	# This only works if the array type has the dimension as the last parameter.
# 	# Uh oh ... how could even know where the dimension parameter is?
#   return freelastparameter(promote_type(arraytype(A), arraytype(B)))
# end


function ==(X::MultiMatrix{LS,RS}, Y::MultiMatrix{LS,RS}) where {LS, RS}
	# Check simple case first: both arrays have the same permutation
	if X.lperm == Y.lperm && X.rperm == Y.rperm
		return X.data == Y.data
	end


	# check whether X,Y have the same elements when permuted
	lperm = Y.lperm[invperm(X.lperm)]
	rperm = Y.rperm[invperm(X.rperm)]

	for (jjX, jjY) in zip(CartesianIndices(raxes(X)), PermIter(raxes(Y), rperm))
		for (iiX, iiY) in zip(CartesianIndices(laxes(X)), PermIter(laxes(Y), lperm))
			if X[iiX,jjX] != Y[iiY,jjY]
				return false
			end
		end
	end
	return true
end


#	square(M)
# If `M` is square (lspaces(M) == rspaces(M) and laxes(M) == raxes(M)), return it.
# If a permuted version of `M` is square, return that.
# Otherwise throw an error.

# This is called if M has the same left and right spaces (possibly in different order)
function square(M::MultiMatrix{LS,LS,T,N,A,NS,NS}) where A<:AbstractArray{T,N} where {LS,T,N,NS}
	if M.lperm == M.rperm
		M
		#laxes(M) == raxes(M) ? M : throw(DimensionMismatch("MultiMatrix is not square"))
		# for i in oneto(NS)
		# 	axes(M,i) == axes(M, i+NS) || throw(DimensionMismatch("MultiMatrix is not square"))
		# end
		# return M
	else
		# permute the dims and try again
		arr = permutedims(M.data, (M.lperm..., (NS .+ M.rperm)...))
		square(MultiMatrix{LS,LS,T,N,A}(arr, oneto(NS), oneto(NS)))
	end
end

# This is the fallback, in the case M's left and right spaces are different
square(M::MultiMatrix{LS,RS}) where {LS,RS} = throw(DimensionMismatch("MultiMatrix is not square"))


# Ensure that M is square in selected spaces; if not possible, throw an error
square(M::MultiMatrix, spaces::Dims) = square(M::MultiMatrix, Val{spaces})
function square(M::MultiMatrix{LS,RS}, ::Type{Val{spaces}}) where {LS,RS} where {spaces}
	S = binteger(IntSet, Val{spaces})
	ldims = M.lperm(findbits(S, LS))
	rdims = M.rperm(findbits(S, RS))
	laxes(M, ldims) == raxes(M, rdims) ? M : throw(DimensionMismatch("MultiMatrix is not square in selected spaces $d"))
end



# Conversions
convert(T, M::MultiMatrix) = convert(T, M.data)


"""
`Matrix(M::MultiMatrix)`

Convert `M` to a `Matrix`. The left (right) dimensions of `M` are reshaped
into the first (second) dimension of the output matrix.
"""
Matrix(M::MultiMatrix) = reshape(M.data, prod(lsize(M)), prod(rsize(M)) )
# function Matrix(M::MultiMatrix{LS,RS,T,N,A,NL,NR}) where {LS,RS,T,N,A,NL,NR}
# 	# sz = size(M)
# 	# l = 1
# 	# r = 1
# 	# for i = oneto(NL)
# 	# 	l *= sz[i]
# 	# end
# 	# for i = tupseq(NL+1, NL+NR)
# 	# 	r *= sz[i]
# 	# end
# 	l = prod(lsize(M))
# 	r = prod(rsize(M))
# 	reshape(M.data, l, r)
# end

# Array access
# If accessing a single element, return that element.
# Otherwise (i.e. if requesting a range) return a MultiMatrix.
getindex(M::MultiMatrix, i::Vararg{Union{Integer, CartesianIndex}}) = getindex(M.data, i...)
getindex(M::MultiMatrix, i...) = MultiMatrix(getindex(M.data, i...))

setindex!(M::MultiMatrix, i...) = setindex!(M.data, i...)


reshape(M::MultiMatrix, shape::Dims) = MultiMatrix(reshape(M.data, shape), spaces(M); checkspaces = false)
permutedims(M::MultiMatrix, ord) = MultiMatrix(permutedims(M.data, ord), spaces(M); checkspaces = false)


# TODO: Make a lazy wrapper, just like Base does.
# Note, "partial adjoint" doesn't really make sense.
function adjoint(M::MultiMatrix)
	n = nspaces(M)
	# adjoint is called element-by-element (i.e. it recurses as required)
	return MultiMatrix(permutedims(adjoint.(M.data), [n+1:2*n; 1:n]), Val(spaces(M)); checkspaces = false)
end


#Swap left and right dimensions of a MultiMatrix
function transpose(M::MultiMatrix)
	n = nspaces(M)
	return MultiMatrix(permutedims(M.data, [n+1:2*n; 1:n]), Val(spaces(M)); checkspaces = false)
end


##------------------------
# The following functions take space labels (not dimensions, or indices of spaces)
# as arguments.

# Partial transpose
function transpose(M::MultiMatrix, ts::Dims)
	S = spaces(M)
	n = nspaces(M)
	#d = findin(S, ts)	#find elements of S in ts
	order = MVector{2*n, Int}(undef)
	is_tdim = in.(S, Ref(ts))
	for i = 1:n
		#if d[i] > 0
		if is_tdim[i]
			order[i] = i+n
			order[i+n] = i
		else
			order[i] = i
			order[i+n] = i+n
		end
	end
	return MultiMatrix(permutedims(M.data, order), S; checkspaces = false)
end

# Attempt to make the compiler infer the permutation order
# function transpose2(M::MultiMatrix, ts::Dims)
# 	S = spaces(M)
# 	order = get_transpose_order(Val(S), Val(ts))
# 	return MultiMatrix(permutedims(M.data, order), S; checkspaces = false)
# end
#
# @generated function get_transpose_order(::Val{S}, ::Val{ts}) where {S,ts}
# 	n = length(S)
# 	#d = findin(S, ts)	#find elements of S in ts
# 	order = MVector{2*n, Int}(undef)
# 	is_tdim = in.(S, Ref(ts))
# 	for i = 1:n
# 		#if d[i] > 0
# 		if is_tdim[i]
# 			order[i] = i+n
# 			order[i+n] = i
# 		else
# 			order[i] = i
# 			order[i+n] = i+n
# 		end
# 	end
# 	return order
# end


# @generated function shared_spaces(A::MultiMatrix{TA,SA}, B::MultiMatrix{TB,SB}) where {TA,SA} where {TB,SB}
# 	t = tuple(Iterators.filter(i-> in(i, SA), SB)...)
# 	return :( $t )
# end



"""
	tr(A)

Returns the trace of a square `MultiMatrix` `A: it contracts each left
space with the corresponding right space and returns a scalar.

	tr(A, spaces = s)

Trace over the the indicated spaces, returning another `MultiMatrix`.
"""
function tr(M::MultiMatrix)
	return tr(Matrix(square(M)))
end

# Partial trace.  It's not donvenient to use a keyword because it clobbers the method
# that doesn't have a spaces argument.
tr(M::MultiMatrix, space::Integer) = tr(M, Val{(space,)})
tr(M::MultiMatrix, spaces::Dims) = tr(M, Val{spaces})

function tr(M::MultiMatrix{LS,RS}, ::Type{Val{tspaces}}) where {LS,RS,tspaces}
	# Int representation of traced spaces
	TS = binteger(IntSet, Val{tspaces})

	# Int representation of kept spaces
	KLS = ~TS & LS
	KRS = ~TS & RS

	# find dims to be traced and ktp

	ltdims = M.lperm[findbits(TS, LS)]
	rtdims = M.rperm[findbits(TS, RS)] .+ nlspaces(M)

	lkdims = M.lperm[findbits(KLS, LS)]
	rkdims = M.rperm[findbits(KRS, RS)] .+ nlspaces(M)

	NL = length(lkdims)
	NR = length(rkdims)
	N = NL + NR
	sz = size(M.data)
	R = similar(M.data, (sz[lkdims]..., sz[rkdims]...))
	# Atype = arraytype(M).name.wrapper
	# R = Atype{eltype(M),N}(undef, sz[lkdims]..., sz[rkdims]...)
	trace!(1, M.data, :N, 0, R, lkdims, rkdims, ltdims, rtdims)
	return MultiMatrix{KLS, KRS, eltype(R), N, typeof(R)}(R, oneto(NL), oneto(NR))
	# function barrier
	#return tr_dims(M, KLS, KRS, ltdims, rtdims, lkdims, rkdims)
end

function tr_dims(M::MultiMatrix, KLS, KRS, ltdims, rtdims, lkdims, rkdims)
	lspaces = lspaces(M)[lkdims]
	rspaces = rspaces(M)[rkdims]
	N = length(lkdims) + length(rkdims)
	sz = size(M.data)
	Atype = arraytype(M).name.wrapper
	R = Atype{eltype(M),N}(undef, sz[kldims]..., sz[krdims]...)
	trace!(1, M.data, :N, 0, R, lkdims, rkdims, ltdims, rtdims)
	return MultiMatrix
end

# Attempts to use compile-term inference.  When it works, it's better, but when it fails,
# it is much worse.
#
# # How to make kdims and tdims evalute at comile time?
# # If we input ::Val(ts), we can make tr a generated function.
# function tr(M::MultiMatrix{T,S,A}, ts::Dims) where {T,S,A}
# 	# is_tdim = in.(S, Ref(ts))
# 	# dims = oneto(length(S))
# 	# tdims = select(dims, is_tdim)
# 	# kdims = deleteat(dims, is_tdim)
# 	(tdims, kdims) = _get_tr_dims(M, Val(ts))
# 	#(tdims, kdims)
# 	return tr_dims(M, tdims, kdims)
# end
#
#
# function tr(M::MultiMatrix{T,S,A}, ::Val{ts}) where {T,S,A} where {ts}
# 	# is_tdim = in.(S, Ref(ts))
# 	# dims = oneto(length(S))
# 	# tdims = select(dims, is_tdim)
# 	# kdims = deleteat(dims, is_tdim)
# 	(tdims, kdims) = _get_tr_dims(M, Val(ts))
# 	#(tdims, kdims)
# 	return tr_dims(M, tdims, kdims)
# end
#
# @generated function _get_tr_dims(M::MultiMatrix{T,S}, ::Val{ts}) where {T,S} where {ts}
# 	is_tdim = collect(map(s->in(s, ts), S))
# 	dims = ntuple(identity, length(S))  #oneto(length(S))
# 	tdims = dims[is_tdim]	#select(dims, is_tdim)
# 	kdims = dims[(!).(is_tdim)]	#deleteat(dims, is_tdim)
# 	return :( ($tdims, $kdims) )
# end





#
# Operations with UniformScaling
#

function +(M::MultiMatrix{S,TM} where {S}, II::UniformScaling{TI}) where {TM,TI<:Number}
	chk_square(M)
	R = LinearAlgebra.copy_oftype(M.data, promote_op(+,TM,TI))
	for ci in CartesianIndices(lsize(M))
		R[ci,ci] += II.λ
	end
	return MultiMatrix(R, Val(spaces(M)); checkspaces = false)
end
(+)(II::UniformScaling, M::MultiMatrix) = M + II

function -(M::MultiMatrix{S,TM} where {S}, II::UniformScaling{TI}) where {TM,TI}
	chk_square(M)
	R = LinearAlgebra.copy_oftype(M.data, promote_(-,TM,TI))
	for ci in CartesianIndices(lsize(M))
		R[ci,ci] -= II.λ
	end
	return MultiMatrix(R, Val(spaces(M)); checkspaces = false)
end

function -(II::UniformScaling{TI}, M::MultiMatrix{S,TM} where {S}) where {TM,TI<:Number}
	chk_square(M)
	R = LinearAlgebra.copy_oftype(M.data, promote_op(-,TI,TM))
	for ci in CartesianIndices(lsize(M))
		R[ci,ci] = II.λ - R[ci,ci]
	end
	return MultiMatrix(R, Val(spaces(M)); checkspaces = false)
end

*(M::MultiMatrix, II::UniformScaling) = M * II.λ
*(II::UniformScaling, M::MultiMatrix) = II.λ * M



#
# Arithmetic operations
#
-(M::MultiMatrix) = MultiMatrix(-M.data, Val(spaces(M)); checkspaces = false)

# Fallback methods (when x is not an abstract array or number)
*(M::MultiMatrix, x) = MultiMatrix(M.data * x, Val(spaces(M)); checkspaces = false)
*(x, M::MultiMatrix) = MultiMatrix(x * M.data, Val(spaces(M)); checkspaces = false)
/(M::MultiMatrix, x) = MultiMatrix(M.data / x, Val(spaces(M)); checkspaces = false)

*(M::MultiMatrix, x::Number) = MultiMatrix(M.data * x, Val(spaces(M)); checkspaces = false)
*(x::Number, M::MultiMatrix) = MultiMatrix(x * M.data, Val(spaces(M)); checkspaces = false)
/(M::MultiMatrix, x::Number) = MultiMatrix(M.data / x, Val(spaces(M)); checkspaces = false)

# TODO:  Handle M .+ x and M .- x so that it returns a MultiMatrix



function +(A::MultiMatrix, B::MultiMatrix)
	if spaces(A) == spaces(B)
		axes(A) == axes(B) || throw(DimensionMismatch("To add MultiMatrices on the same spaces, they must have the same axes; got axes(A) = $(axes(A)), axes(B) = $(axes(B))"))
		return MultiMatrix(A.data + B.data, Val(spaces(A)); checkspaces = false)
	# TODO: elseif spaces are the same, but in different order ...
	else
		error("Not implemented")
		return add_different(A, B)
	end
end

#
# function add_different(A::MultiMatrix, B::MultiMatrix)
# 	SA = spaces(A)
# 	SB = spaces(B)
# 	(SRv, AinR, BinR) = indexed_union(SA, SB)
#
# 	SR = tuple(SRv)
#
# 	A = promote_type(arraytype(A), arraytype(B))
#
# 	C = MultiMatrix{SR, eltype(A), A
#
# 	return _add_different(A, B, SR, AinR, BinR)
# end
#
# function _add_different(A::MultiMatrix, B::MultiMatrix, SR,
# 	# AinC = indices of A in C
# 	# BinC = inices of B in C
#
# 	#for iC in CartesianIndices(
# 	#	for jC in CartisianIndices
# 	#C[iC] = A[iC[AinC]] + B[iC[BinC]]
# 	#
# 	#
# end


function -(A::MultiMatrix, B::MultiMatrix)
	spaces(A) == spaces(B) || error("To subtract MultiMatrices, they must have the same spaces in the same order")
	axes(A) == axes(B) || throw(DimensionMismatch("To subtract MultiMatrices, they must have the same axes; got axes(A) = $(axes(A)), axes(B) = $(axes(B))"))
	return MultiMatrix(A.data - B.data, Val(spaces(A)); checkspaces = false)
end

# Matrix multiplication.  These methods are actually quite fast -- about as fast as the
# core matrix multiplication. We appear to incur very little overhead.
"""
`M*X` where `M` is a MultiMatrix and `X` is an `AbstractArray` contracts the right
dimensions of `M` with dimensions `spaces(M)` of `X`.  The result is an array of type
similar to `X`, whose size along the contracted dimensions is `lsize(M)` and whose size in
the uncontracted dimensions is that of `X`.

`X*M` is similar, except the left dimensions of `M` are contracted against `X`, and the
size of the result depends on the `rsize(M)`.
"""
*(M::MultiMatrix, A::AbstractArray{TA}) where {TA} = _mult_MA(M, A)
*(M::MultiMatrix, A::AbstractArray{TA,1}) where {TA} = _mult_MA(M, A)
*(M::MultiMatrix, A::AbstractArray{TA,2}) where {TA} = _mult_MA(M, A)

function _mult_MA(M::MultiMatrix, A::AbstractArray{TA}) where {TA}
	n = nspaces(M)
	S = spaces(M)
	raxes(M) == axes(A, S) || throw(DimensionMismatch("raxes(M) must equal axes(B, spaces(A))"))

	nR = max(ndims(A), maximum(S))
	kdimsA = deleteat(oneto(nR), S)

	szR = MVector(size(A, oneto(nR)))
	lszM = lsize(M)
	for i = 1:n
		szR[S[i]] = lszM[i]
	end
	TR = promote_op(*, eltype(M), TA)
	R = similar(A, TR, Tuple(szR))
	contract!(one(eltype(M)), M.data, :N, A, :N, zero(TR), R, oneto(n), tupseq(n+1,2*n), kdimsA, S, invperm((S...,kdimsA...)))
	return R
end
#*(M::MultiMatrix, A::AbstractArray{TA,1}) where {TA} =

*(A::AbstractArray{TA}, M::MultiMatrix) where {TA} = _mult_AM(A, M)
*(A::AbstractArray{TA,1}, M::MultiMatrix) where {TA} = _mult_AM(A, M)
*(A::AbstractArray{TA,2}, M::MultiMatrix) where {TA} = _mult_AM(A, M)

# This is almost identical to the M*A version.
function _mult_AM(A::AbstractArray{TA}, M::MultiMatrix) where {TA}
	n = nspaces(M)
	S = spaces(M)
	laxes(M) == axes(A, S) || throw(DimensionMismatch("axes(A, spaces(B)) must equal laxes(B)"))

	nR = max(ndims(A), maximum(S))
	kdimsA = deleteat(oneto(nR), S)

	szR = MVector(size(A, oneto(nR)))
	rszM = rsize(M)
	for i = 1:n
		szR[S[i]] = rszM[i]
	end
	TR = promote_op(*, eltype(M), TA)
	R = similar(A, TR, Tuple(szR))
	contract!(one(eltype(M)), M.data, :N, A, :N, zero(TR), R, tupseq(n+1,2*n), oneto(n), kdimsA, S, invperm((S...,kdimsA...)))
	return R
end


"""
`A*B` where `A` and `B` are MultiMatrices.
"""
function *(A::MultiMatrix, B::MultiMatrix)
	Atype = arraytype(A).name.wrapper
	Btype = arraytype(B).name.wrapper
	Atype == Btype || error("To multiply MultiMatrices, the underlying array types must be the same.  Had types $Atype and $Btype")
	lszA = lsize(A)
	rszA = rsize(A)
	lszB = lsize(B)
	rszB = rsize(B)

	nA = nspaces(A)
	nB = nspaces(B)
	SA = spaces(A)
	SB = spaces(B)

	if SA == SB
		# Simple case:  spaces(A) = spaces(B)
		raxes(A) == laxes(B) || throw(DimensionMismatch("raxes(A) = $(raxes(A)) must equal laxes(B) = $(laxes(B))"))
		R = reshape(Matrix(A) * Matrix(B), tcat(lszA, rszB))
		return MultiMatrix(R, Val(SA); checkspaces = false)
	else
		# General case
		(tdimsA, tdimsB, odimsA, odimsB, dimsR, SR) = get_mult_dims(Val(spaces(A)), Val(spaces(B)))
		# println("tdims A,B = $tdimsA <--> $tdimsB")
		# println("odimsA = $odimsA")
		# println("odimsB = $odimsB")
		# println("dimsR = $dimsR")
		# println("spacesR = $SR")

		axes(A, tdimsA) == axes(B, tdimsB) || throw(DimensionMismatch("raxes(A) must equal laxes(B) on spaces common to A,B"))

		szAB = tcat(size(A, odimsA), size(B, odimsB))
		szR = szAB[dimsR]
		#println("sizeR = $szR")
		#TR = promote_type(TA, TB)
		#szR = (lszA[kdimsA]..., lszB[kdimsB])
		TR = promote_op(*, eltype(A), eltype(B))
		R = Atype{TR}(undef, szR)
		contract!(one(eltype(A)), A.data, :N, B.data, :N, zero(TR), R, odimsA, tdimsA, odimsB, tdimsB, dimsR)
		return MultiMatrix(R, Val(SR); checkspaces = false)
		#return nothing
	end
end



@generated function get_mult_dims(::Val{SA}, ::Val{SB}) where {SA,SB}
	# Example:
	#    C[o1,o2,o3,o4; o1_,o2_,o3_,o4_] = A[o1, o2, o3; o1_, c2_, c3_] * B[c2, c3, o4; o2_, o3_, o4_]
	# itsa, itsb = indices of contracted spaces of A, B
	# iosa, iosb = indices of open spaces of A, B
	# odimsA = 1:nA, nA+iosa
	# tdimsA = nA + itsa
	# tdimsB = itsb
	# odimsB = iosb, nB+(1:nB)

	nA = length(SA)
	nB = length(SB)
	tsa_ = MVector{nA,Int}(undef)
	tsb_ = MVector{nB,Int}(undef)
	osa_mask = @MVector ones(Bool, nA)
	osb_mask = @MVector ones(Bool, nB)

	nt = 0
	for i = 1:nA
		for j = 1:nB
	 		if SA[i] == SB[j]
				nt += 1
	 			tsa_[nt] = i
	 			tsb_[nt] = j
				osa_mask[i] = false
				osb_mask[j] = false
	 		end
	 	end
	end


	itsa = Tuple(tsa_)[oneto(nt)]
	itsb = Tuple(tsb_)[oneto(nt)]
	iosa = oneto(nA)[osa_mask]
	iosb = oneto(nB)[osb_mask]

	nosA = nA - nt		# number of open spaces of A
	nosB	= nB - nt		# number of open spaces of B
	nR = nA + nB - nt	# = nA + noB = nB + noA)
	tdimsA = nA .+ itsa
	tdimsB = itsb
	odimsA = tcat(oneto(nA), (nA .+ iosa))
	odimsB = tcat(iosb, tupseq(nB+1, 2*nB))

	# open spaces of A, common spaces, open spaces of B
	nodA = nA + nosA
	SR = tcat(SA[itsa], SA[iosa], SB[iosb])
	dimsR = tcat(itsa, iosa, nodA .+ oneto(nosB), (nodA + nosB) .+ itsb, tupseq(nA+1, nA+nosA), (nodA + nosB) .+ iosb)
	return :( ($tdimsA, $tdimsB, $odimsA, $odimsB, $dimsR, $SR) )
end


^(M::MultiMatrix, x::Number) = MultiMatrix(reshape(Matrix(M)^x, size(M)), Val(spaces(M)); checkspaces = false)
^(M::MultiMatrix, x::Integer) = MultiMatrix(reshape(Matrix(M)^x, size(M)), Val(spaces(M)); checkspaces = false)

#
# Analytic matrix functions
#

for f in [:inv, :sqrt, :exp, :log, :sin, :cos, :tan, :sinh, :cosh, :tanh]
	@eval function $f(M::MultiMatrix)
			chk_square(M)
			MultiMatrix(reshape($f(Matrix(M)), size(M)), Val(spaces(M)); checkspaces = false)
		end
end


#
# Linear-algebra functions
#
det(M::MultiMatrix) = begin chk_square(M); det(Matrix(M)); end
opnorm(M::MultiMatrix, args...) = begin chk_square(M); opnorm(Matrix(M), args...); end

eigvals(M::MultiMatrix, args...) = begin chk_square(M); eigvals(Matrix(M), args...); end
svdvals(M::MultiMatrix, args...) = svdvals(Matrix(M), args...)


struct MultiMatrixStyle{S,A,N} <: Broadcast.AbstractArrayStyle{N} end
MultiMatrixStyle{S,A,N}(::Val{N}) where {S,A,N} = MultiMatrixStyle{S,A,N}()

similar(bc::Broadcasted{MMS}, ::Type{T}) where {MMS<:MultiMatrixStyle{S,A,N}} where {S,N} where {A<:AbstractArray} where {T} = similar(MultiMatrix{S,T,N,A}, axes(bc))

BroadcastStyle(::Type{MultiMatrix{S,T,N,A}}) where {S,T,N,A} = MultiMatrixStyle{S,A,N}()
BroadcastStyle(::Type{MultiMatrix{S,T1,N,A1}}, ::Type{MultiMatrix{S,T2,N,A2}})  where {A1<:AbstractArray{T1,N}} where {A2<:AbstractArray{T2,N}} where {S,N} where {T1,T2} = MultiMatrixStyle{S, promote_type(A1,A2), N}()
BroadcastStyle(::Type{<:MultiMatrix}, ::Type{<:MultiMatrix}) = error("To be broadcasted, MultiMatrices must have the same dimensions and the same spaces in the same order")

# BroadcastStyle(::Type{BitString{L,N}}) where {L,N} = BitStringStyle{L,N}()
# BroadcastStyle(::BitStringStyle, t::Broadcast.DefaultArrayStyle{N}) where {N} = Broadcast.DefaultArrayStyle{max(1,N)}()


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
