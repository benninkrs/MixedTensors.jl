"""
	MultiMatrices (module)

Multimatrices generalize matrices to tensor product spaces. A `MultiMatrix` of
size (l1,…,ln,r1,…,rn) represents a linear map between a "left"
vector space L of size (l1,...,ln) and a "right" vector space R of size
(r1,...,rn). (In the language of tensors, a multimatrix is a tensor with an equal
number of "up" and "down" indices.)
"""

# TODO: Do we really need to parameterize by the spaces?
#	- To do the space math efficiently, we just need to know the tuple lengths
#	- However, the output of get_mult_dims is (amazinagly!) inferred (at least for low-
#		dimensional matrices).  If we got rid of the space parameter we would gain type
#		stability on more constructors, but lose type stability on multiplication (wouldn't
#		be able to infer the dimensionality of the result)
# TODO: Figure out why broadcasting is slow
# TODO: Parameterize by Tuple{spaces} instead of spaces -- infers better? (a la how StaticArrays does it)
#	- No, it doesn't help with constructors like M((5,3,6)).
#	- To get inferred it Would need to be M(Tuple{5,3,6}) which is almost as bad as M(Val((5,3,6))).
# TODO: Use generated functions to speed up dimension-mangling
# TODO: Use dispatch to separate out A*B with same spaces
# TODO: Extend addition, subtraction to allow spaces to be in different order
# TODO: Extend * to form lazy outer products
# TODO: Support +,- for dissimilar spaces?  Perhaps lazy structures?
# TODO: Check validity of dims in lsize, rsize, laxes, raxes
# TODO: Use Strided.jl instead of TensorOperations?
# TODO: Support in-place operations?
# TODO: Generalize to different in spaces and out spaces?

module MultiMatrices

export MultiMatrix, lsize, rsize, spaces, nspaces, arraytype, laxes, raxes

using MiscUtils
using SuperTuples
using StaticArrays
using LinearAlgebra
using TensorOperations: trace!, contract!
using Base.Broadcast: Broadcasted, BroadcastStyle
#using PermutedIteration

using Base: promote_op

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


"""
`MultiMatrix(A::AbstractArray, S::Dims)` creates a multimatrix from array `A` acting on
spaces `S`. The elements of `S` must be unique `ndims(A)` must equal `2*length(S)'.
"""
struct MultiMatrix{S, T, N, A<:AbstractArray{T,N}} <: AbstractArray{T,N}
	data::A
	# Constructor with input validation
	function MultiMatrix{S,T,N,A}(data::A; checkspaces = true) where {A<:AbstractArray{T,N}} where {T,N,S}
		if N != 2*length(S)
			error("ndims(A) must be twice the number of specified spaces.")
		end
		if checkspaces && !allunique(S)
			error("Spaces must be unique")
		end
		return new{S,T,N,A}(data)
	end
end

# Convenience constructors

# Construct from array with default spaces
function MultiMatrix(arr::A) where A<:AbstractArray{T,N} where {T,N}
	iseven(N) || error("Source array must have an even number of dimensions")
	MultiMatrix{oneto(N>>>1),T,N,A}(arr; checkspaces = false)
end

# Construct from array with custom spaces
function MultiMatrix(arr::A, S::Dims; checkspaces = true) where A<:AbstractArray{T,N} where {T,N}
	MultiMatrix{S,T,N,A}(arr; checkspaces = checkspaces)
end

# construct from array with custom spaces (type-inferrable)
function MultiMatrix(arr::A, ::Val{S}; checkspaces = true) where A<:AbstractArray{T,N} where {T,N,S}
	MultiMatrix{S,T,N,A}(arr; checkspaces = checkspaces)
end


# Reconstruct with different spaces
"""
`(M::MultiMatrix)(S::Dims)` or `(M::MultiMatrix)(S::Int...)`

Create a MultiMatrix with the same data as `M` but acting on spaces `S`.
"""
(M::MultiMatrix)(spaces::Vararg{Int64}) = M(spaces)
#(M::MultiMatrix)(spaces::Tuple{Vararg{Int64}}) = MultiMatrix(M.data, spaces)
(M::MultiMatrix{S,T,N,A})(spaces::Tuple{Vararg{Int64}}) where {S,T,N,A<:AbstractArray{T,N}} = MultiMatrix{spaces,T,N,A}(M.data)


similar(M::MultiMatrix) = MultiMatrix(similar(M.data), Val(spaces(M)); checkspaces = false)
similar(M::MultiMatrix, args...) = MultiMatrix(similar(M.data, args...), Val(spaces(M)); checkspaces = false)

# Construct undef based on type and size
const Shape = Tuple{Union{Integer, Base.OneTo},Vararg{Union{Integer, Base.OneTo}}}

similar(::Type{M}, shape::Shape) where {M<:MultiMatrix{S,T,N,A}} where {S} where {A<:AbstractArray{T,N}} where {T,N} = M(similar(A, shape))

#-------------
# The following methods all refer on the "native" dimensions

# Size and shape
ndims(M::MultiMatrix) = ndims(M.data)
spaces(M::MultiMatrix{S}) where {S} = S
nspaces(M::MultiMatrix{S}) where {S} = length(S)
length(M::MultiMatrix) = length(M.data)

size(M::MultiMatrix) = size(M.data)
size(M::MultiMatrix, dim) = size(M.data, dim)
size(M::MultiMatrix, dims::Iterable) = size(M.data, dims)

lsize(M::MultiMatrix) =  ntuple(d -> size(M.data, d), nspaces(M))
lsize(M::MultiMatrix, dim) =  size(M.data, dim)
lsize(M::MultiMatrix, dim::Iterable) =  map(d -> size(M.data, d), dims)

rsize(M::MultiMatrix) = ntuple(d -> size(M.data, d+nspaces(M)), nspaces(M))
rsize(M::MultiMatrix, dim) =  size(M.data, dim + nspaces(M))
rsize(M::MultiMatrix, dim::Iterable) =  map(d -> size(M.data, d + nspaces(M)), dims)

axes(M::MultiMatrix) = axes(M.data)
axes(M::MultiMatrix, dim) = axes(M.data, dim)
axes(M::MultiMatrix, dims::Iterable) = map(d->axes(M.data, d), dims)


"""
`laxes(M)` left axes of `M`.
"""
#laxes(M::MultiMatrix) = ntuple(d -> axes(M.data, d), nspaces(M))		# inexplicably, this doesn't infer even though raxes(M) does
laxes(M::MultiMatrix) = map(d -> axes(M.data, d), oneto(nspaces(M)))
laxes(M::MultiMatrix, dim) = axes(M.data, dim)
laxes(M::MultiMatrix, dims::Iterable) = map(d -> axes(M.data, d), dims)

"""
`raxes(M)` right axes of `M`.
"""
raxes(M::MultiMatrix) = ntuple(d -> axes(M.data, d+nspaces(M)), nspaces(M))
raxes(M::MultiMatrix, dim) = axes(M.data, dim + nspaces(M))
raxes(M::MultiMatrix, dims::Iterable) = map(d -> axes(M.data, d + nspaces(M)), dims)


arraytype(::MultiMatrix{S,T,N,A} where {S,T,N}) where A = A


function ==(X::MultiMatrix, Y::MultiMatrix)
	# Check simple case first, SX == SY
	SX = spaces(X)
	SY = spaces(Y)
	if SX == SY
		return X.data == Y.data
	end

	# check whether X,Y have the same spaces (in any order)
	iX = sortperm(SX)
	iY = sortperm(SY)
	(SX[iX] != SY[iY]) && return false

	# check whether X,Y have the same (sorted) axes
	if laxes(X, iX) != laxes(Y, iY) || raxes(X, iX) != raxes(Y, iY)
		return false
	end


	# check whether X,Y have the same elements when permuted
	for (jjX, jjY) in zip(PermIter(raxes(X), iX), PermIter(raxes(Y), iY))
		for (iiX, iiY) in zip(PermIter(laxes(X), iX), PermIter(laxes(Y), iY))
			if X[iiX,jjX] != Y[iiY,jjY]
				return false
			end
		end
	end
	return true

	# Slightly slower -- applies permutation to each index
	# for jj in CartesianIndices(raxes(X, iX))
	# 	pjx = invpermute(Tuple(jj), iX)
	# 	pjy = invpermute(Tuple(jj), iY)
	# 	for ii in CartesianIndices(laxes(X, iX))
	# 		ijX = vcat(invpermute(Tuple(ii), iX), pjx)
	# 		ijY = vcat(invpermute(Tuple(ii), iY), pjy)
	# 		if X[ijX...] != Y[ijY...]
	# 			return false
	# 		end
	# 	end
	# end
	return true
end



"""
`chk_square(M::MultiMatrix)`

Thow an error if `M` is not "square" (`laxes(M) == raxes(M)`).
"""
chk_square(M::MultiMatrix) = laxes(M) == raxes(M) ? M : throw(DimensionMismatch("MultiMatrix is not square: laxes = $(laxes(M)), raxes = $(raxes(M))"))
chk_square(M::MultiMatrix, d::Dims) = laxes(M, d) == raxes(M, d) ? M : throw(DimensionMismatch("MultiMatrix is not square in selected dimensions $d"))



# Conversions
convert(T, M::MultiMatrix) = convert(T, M.data)


"""
`Matrix(M::MultiMatrix)`

Convert `M` to a `Matrix`. The left (right) dimensions of `M` are reshaped
into the first (second) dimension of the output matrix.
"""
Matrix(M::MultiMatrix) = reshape(M.data, prod(lsize(M)), prod(rsize(M)) )



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
# The following functions involve explicit spaces. Therefore, these refer to actual
# spaces not relative.

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
`tr(A)` returns the trace of a `MultiMatrix` `A`, i.e. it contracts each left
dimension with the corresponding right dimension and returns a scalar.

`tr(A, spaces)` traces out the indicated spaces, returning another `MultiMatrix`
(even if all the spaces are traced).
"""
function tr(M::MultiMatrix)
	chk_square(M)
	return tr(Matrix(M))
end

# Partial trace
tr(M::MultiMatrix, i::Integer) = tr(M, (i,))

function tr(M::MultiMatrix, ts::Dims)
	S = spaces(M)
	n = nspaces(M)
	is_tdim = in.(S, Ref(ts))
	dims = oneto(n)
	tdims = dims[is_tdim]
	kdims = dims[(!).(is_tdim)]
	chk_square(M, tdims)
	return tr_dims(M, tdims, kdims)
end

function tr_dims(M::MultiMatrix, tdims, kdims::Dims)
	n = nspaces(M)
	S = spaces(M)
	K = length(kdims)
	lsz = lsize(M)
	rsz = rsize(M)
	Atype = arraytype(M).name.wrapper
	R = Atype{eltype(M),2*K}(undef, lsz[kdims]..., rsz[kdims]...)
	trace!(1, M.data, :N, 0, R, kdims, kdims+n, tdims, tdims+n)
	return MultiMatrix(R, S[kdims]; checkspaces = false)
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
	spaces(A) == spaces(B) || error("To add MultiMatrices, they must have the same spaces in the same order.")
	axes(A) == axes(B) || throw(DimensionMismatch("To add MultiMatrices, they must have the same axes; got axes(A) = $(axes(A)), axes(B) = $(axes(B))"))
	return MultiMatrix(A.data + B.data, Val(spaces(A)); checkspaces = false)
end

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
