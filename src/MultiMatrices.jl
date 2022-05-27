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

export MultiMatrix, multivector, lsize, rsize, spaces, lspaces, rspaces, nlspaces, nrspaces
#export ldim, rdim

using MiscUtils
using SuperTuples
using StaticArrays
using LinearAlgebra
using TensorOperations: trace!, contract!, tensoradd!
using Base.Broadcast: Broadcasted, BroadcastStyle
# using PermutedIteration

using Base: promote_op

import Base: display, show
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

const Axes{N} = NTuple{N, AbstractUnitRange{<:Integer}}

# Alias for whatever integer type we use to specify a set of vector spaces
# Not to be confused with DataStructures.IntSet or Base.BitSet.
const IntSet = UInt128

"""
	MultiMatrix{LS, RS, T, N, A<:AbstractArray{T,N}}	(struct)

A `MultiMatrix` represents a linear map between products of vector spaces.
`LS` and `RS` are wide unsigned integers whose bits specify the "left" and "right" vector
spaces, respectively.
"""
struct MultiMatrix{LS, RS, T, N, A<:AbstractArray{T,N}, NL, NR} <: AbstractArray{T,N}
	data::A
	ldims::Dims{NL}		# dims of A corresponding to the ordered left spaces
								#	(a permutation of 1:NL)
	rdims::Dims{NR}		# dims of A corresponding to the ordered right spaces
								#	(a permutation of NL+1:NL+NR)

	# Primary inner constructor. Validates inputs
	# TODO - check that ldims and rdims have valid values?
	function MultiMatrix{LS, RS}(data::A, ldims::NTuple{NL,Integer}, rdims::NTuple{NR,Integer}) where {LS, RS, NL, NR} where {A<:AbstractArray{T,N}} where {T,N}
		count_ones(LS) == NL || error("length(ldims) must equal count_ones(LS)")
		count_ones(RS) == NR || error("length(rdims) must equal count_ones(RS)")
		# isperm((ldims...,rdims...)) || error("ldims and rdims must form a permutaiton of the array's dimensions")
		NL + NR == N || error("ndims(A) must be equal the total number of spaces (left + right)")
		return new{LS, RS, T, N, A, NL, NR}(data, ldims, rdims)
	end

	# Shortcut constructor: construct from array, using another MultiMatrix's metadata.
	# By ensuring the array has the right number of dimensions, no input checking is needed.
	function MultiMatrix(arr::A_, M::MultiMatrix{LS,RS,T,N,A,NL,NR}) where {A_ <: AbstractArray{T_,N}} where {T_} where {LS,RS,T,N,A,NL,NR}
		return new{LS,RS,T_,N,A_,NL,NR}(arr, M.ldims, M.rdims)
	end

end

# Convenience constructors

# Construct from array with default spaces
"""
	MultiMatrix(A::AbstractArray)
	MultiMatrix(A::AbstractArray, spaces::Tuple)
	MultiMatrix(A::AbstractArray, lspaces::Tuple, rspaces::Tuple)

Create from array 'A' a MultiMatrix that acts on specified vector spaces.
The elements of `spaces` (or `lspaces` or `rspaces`) must be distinct.
If omitted, `spaces` is taken to be `(1,...,ndims(A)/2)`.
`rspaces` defaults to `lspaces`.
`length(lspaces) + length(rspaces)` must equal `ndims(A)`.

	MultiMatrix(A::AbstractArray, M::MultiMatrix)

uses the spaces from `M`, provided ndims(A) == ndims(M).

 To construct a MultiMatrix with default left spaces and no right spaces, use
 [`multivector(A)`](@ref).
"""
function MultiMatrix(arr::A) where A<:AbstractArray{T,N} where {T,N}
	iseven(N) || error("Source array must have an even number of dimensions")
	M = N >> 1
	LS = IntSet(N-1)		# trick to create 001⋯1 with M ones
	MultiMatrix{LS,LS}(arr, oneto(M), tupseq(M+1,2*M))
end


# Construct from array with custom spaces.
MultiMatrix(arr, spaces::Dims) = MultiMatrix(arr, Val{spaces})
MultiMatrix(arr, lspaces::Dims, rspaces::Dims) = MultiMatrix(arr, Val{lspaces}, Val{rspaces})

# Would it make sense to have a constructor that takes LS,RS parameters explicitly?

# Construct from array with custom spaces - type inferred.
function MultiMatrix(arr::A, ::Type{Val{S}}) where {S} where A<:AbstractArray{T,N} where {T,N}
	perm = sortperm(S)
	SI = binteger(IntSet, Val{S})
	MultiMatrix{SI,SI}(arr, perm, perm .+ length(S))
end

function MultiMatrix(arr::A, ::Type{Val{Lspaces}}, ::Type{Val{Rspaces}}) where {Lspaces, Rspaces} where A<:AbstractArray{T,N} where {T,N}
	ldims = sortperm(Lspaces)
	rdims = sortperm(Rspaces) .+ length(Lspaces)
	LS = binteger(IntSet, Val{Lspaces})
	RS = binteger(IntSet, Val{Rspaces})
	MultiMatrix{LS,RS}(arr, ldims, rdims)
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
(M::MultiMatrix)(::Type{Val{S}}) where {S} = MultiMatrix(M.data, Val{S})
(M::MultiMatrix)(lspaces::Dims, rspaces::Dims) where {N} = MultiMatrix(M.data, Val{lspaces}, Val{rspaces})
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
# ldims(M::MultiMatrix) = M.lperm
# rdims(M::MultiMatrix) = length(M.lperm) + M.rperm

lspace_int(M::MultiMatrix{LS, RS}) where {LS,RS} = LS
rspace_int(M::MultiMatrix{LS, RS}) where {LS,RS} = RS

nlspaces(M::MultiMatrix{LS, RS, T, N, A, NL, NR}) where {LS, RS, T, N, A, NL, NR} = NL
nrspaces(M::MultiMatrix{LS, RS, T, N, A, NL, NR}) where {LS, RS, T, N, A, NL, NR} = NR

# Return the spaces in array order
spaces(M::MultiMatrix) = (lspaces(M), rspaces(M))
lspaces(M::MultiMatrix{LS}) where {LS} = invpermute(findnzbits(LS), M.ldims)
rspaces(M::MultiMatrix{LS, RS, T, N, A, NL, NR}) where {LS, RS, T, N, A, NL, NR} = invpermute(findnzbits(RS), M.rdims .- NL)

length(M::MultiMatrix) = length(M.data)

size(M::MultiMatrix) = size(M.data)
size(M::MultiMatrix, dim) = size(M.data, dim)
size(M::MultiMatrix, dims::Iterable) = size(M.data, dims)

axes(M::MultiMatrix) = axes(M.data)
axes(M::MultiMatrix, dim) = axes(M.data, dim)
axes(M::MultiMatrix, dims::Iterable) = map(d->axes(M.data, d), dims)

lsize(M::MultiMatrix) = ntuple(d -> size(M.data, d), Val(nlspaces(M)))
lsize(M::MultiMatrix, dim) =  size(M.data, dim)
lsize(M::MultiMatrix, dim::Iterable) =  map(d -> size(M.data, d), dims)

rsize(M::MultiMatrix) = ntuple(d -> size(M.data, d+nlspaces(M)), Val(nrspaces(M)))
rsize(M::MultiMatrix, dim) =  size(M.data, dim + nlspaces(M))
rsize(M::MultiMatrix, dim::Iterable) =  map(d -> size(M.data, d + nspaces(M)), dims)

# Not sure laxes and raxes are really necessary
laxes(M::MultiMatrix) = ntuple(d -> axes(M.data, d), Val(nlspaces(M)))
laxes(M::MultiMatrix, dim) = axes(M.data, dim)
laxes(M::MultiMatrix, dims::Iterable) = map(d -> axes(M.data, d), dims)

raxes(M::MultiMatrix) = ntuple(d -> axes(M.data, d+nlspaces(M)), Val(nrspaces(M)))
raxes(M::MultiMatrix, dim) = axes(M.data, dim + nlspaces(M))
raxes(M::MultiMatrix, dims::Iterable) = map(d -> axes(M.data, d + nlspaces(M)), dims)


arraytype(::MultiMatrix{LS,RS,T,N,A} where {LS,RS,T,N}) where A = A

calc_strides(sz::Dims{N}) where {N} = cumprod(ntuple(i -> i==1 ? 1 : sz[i-1], Val(N)))
calc_strides(ax::Axes{N}) where {N} = cumprod(ntuple(i -> i==1 ? 1 : last(ax[i-1]) - first(ax[i-1]) + 1, Val(N)))

@inline calc_index(I::CartesianIndex{N}, strides::NTuple{N,Int}) where {N} = 1 + sum((Tuple(I) .- 1) .* strides)


size2string(d) = isempty(d) ? "0-dim" :
                 length(d) == 1 ? "length-$(d[1])" :
                 join(map(string,d), '×')

display(M::MultiMatrix) = show(M)

function show(io::IO, M::MultiMatrix)
	# print as usual
	print(io, size2string(size(M.data)), " ")
	print(io, "MultiMatrix{", lspaces(M), "⟷", rspaces(M), ", ", arraytype(M), "}")

	if !isempty(M.data)
		println(io, ':')
		Base.print_array(IOContext(io, :compact => true, :typeinfo => eltype(M.data)), M.data)
	end
end


# # Return the type of an array
# function promote_arraytype(A::MultiMatrix, B::MultiMatrix)
# 	# This only works if the array type has the dimension as the last parameter.
# 	# Uh oh ... how could even know where the dimension parameter is?
#   return freelastparameter(promote_type(arraytype(A), arraytype(B)))
# end
# X and Y have the same spaces
function ==(X::MultiMatrix{LS,RS}, Y::MultiMatrix{LS,RS}) where {LS, RS}
	# Check whether both arrays have the same permutation
	if X.ldims == Y.ldims && X.rdims == Y.rdims
		return X.data == Y.data
	end

	# check whether X,Y have the same elements when permuted

	# I tried combining the permutations so that only a single linear index was calculated,
	# but surprisingly that ended up being slower
	Xdims = (X.ldims..., X.rdims...)
	Ydims = (Y.ldims..., Y.rdims...)
	ax = axes(X)[Xdims]

	Xstrides = calc_strides(axes(X))[Xdims]
	Ystrides = calc_strides(axes(Y))[Ydims]

	for ci in CartesianIndices(ax)
		iX = calc_index(ci, Xstrides)
		iY = calc_index(ci, Ystrides)
		if X[iX] != Y[iY]
			return false
		end
	end
	return true

end

# Fallback when X,Y have different spaces
==(X::MultiMatrix, Y::MultiMatrix) = false


# Internal functions to make ensure a MultiMatrix is "square".
# A MultiMatrix M is *square* if it has the same left and right spaces (in any order)
# and the corresponding left and right axes are the same.
# It is "proper" square if it is square and the left and right spaces are in the same order.
#
# square(M) returns the square version of M (either M or a permutation of M) if it exists;
# otherwise throws an error

@inline function square(M::MultiMatrix{LS,LS}) where {LS}
	axes(M, M.ldims) == axes(M, M.rdims) ? M : throw(DimensionMismatch("MultiMatrix is not square"))
end

# This is the fallback, in the case M's left and right spaces are different
@inline square(M::MultiMatrix{LS,RS}) where {LS,RS} = throw(DimensionMismatch("MultiMatrix is not square"))


# This is called if M has the same left and right spaces (possibly in different order)
@inline function proper_square(M::MultiMatrix{LS,LS,T,N,A,NS,NS}) where A<:AbstractArray{T,N} where {LS,T,N,NS}
	if M.ldims == M.rdims .- NS
		laxes(M) == raxes(M) ? M : throw(DimensionMismatch("MultiMatrix is not square"))
	else
		# permute the dims and try again
		arr = permutedims(M.data, (M.ldims..., M.rdims...))
		square(MultiMatrix{LS,LS}(arr, oneto(NS), tupseq(NS+1, 2*NS)))
	end
end

# This is the fallback, in the case M's left and right spaces are different
@inline proper_square(M::MultiMatrix{LS,RS}) where {LS,RS} = throw(DimensionMismatch("MultiMatrix is not square"))


# # Ensure that M is square in selected spaces; if not possible, throw an error
# square(M::MultiMatrix, spaces::Dims) = square(M::MultiMatrix, Val{spaces})
# function square(M::MultiMatrix{LS,RS}, ::Type{Val{spaces}}) where {LS,RS} where {spaces}
# 	S = binteger(IntSet, Val{spaces})
#
# 	# TODO - update
# 	# ldims = M.lperm(findnzbits(S, LS))
# 	# rdims = M.rperm(findnzbits(S, RS))
# 	axes(M, M.ldims) == axes(M, M.rdims) ? M : throw(DimensionMismatch("MultiMatrix is not square in selected spaces $d"))
# end



# Conversions
convert(T, M::MultiMatrix) = convert(T, M.data)


"""
`Matrix(M::MultiMatrix)`

Convert `M` to a `Matrix`. The left (right) dimensions of `M` are reshaped
into the first (second) dimension of the output matrix.
"""
Matrix(M::MultiMatrix) = reshape(M.data, prod(lsize(M)), prod(rsize(M)) )


# ------------------------
# Array access


# If accessing a single element, return that element.
getindex(M::MultiMatrix, i::Vararg{Union{Integer, CartesianIndex}}) = getindex(M.data, i...)


# Otherwise (i.e. if requesting a range) return a MultiMatrix.
function getindex(M::MultiMatrix, idx...)
	I = to_indices(M.data, idx)
	NL = nlspaces(M)
	NR = nrspaces(M)
	lkeep = map(i -> !isa(i, Integer), I[1:NL])
	rkeep = map(i -> !isa(i, Integer), I[NL+1:NL+NR])

	(LS_, RS_) = getindex_spaces(M, idx)
	NL_ = count_ones(LS_)
	NR_ = count_ones(RS_)

	ldims__ = _getindex_dims(lspace_int(M), M.ldims, lkeep)
	ldims_ = ntuple(i->ldims__[i], Val(NL_))

	rdims__ = _getindex_dims(rspace_int(M), M.rdims, rkeep, NL, NL_)
	rdims_ = ntuple(i->rdims__[i], Val(NR_))
	MultiMatrix{LS_,RS_}(getindex(M.data, idx...), ldims_, rdims_)
end

setindex!(M::MultiMatrix, i...) = setindex!(M.data, i...)

# Helper functions

# Cacluate the spaces that remain when indexing a MultiMatrix
function getindex_spaces(M::MultiMatrix, idx)
	NL = nlspaces(M)
	NR = nrspaces(M)
	I =  to_indices(M.data, idx)
	length(I) == ndims(M) || error("Wrong number of indices")
	lidx = ntuple(i->idx[i], Val(NL))
	ridx = ntuple(i->idx[i+NL], Val(NR))

	LS_ = _getindex_spaces(Val{lspace_int(M)}, lidx...)
	RS_ = _getindex_spaces(Val{rspace_int(M)}, ridx...)
	(LS_, RS_)
end

_getindex_spaces(::Type{Val{S}}) where {S} = IntSet(0)

function _getindex_spaces(::Type{Val{S}}, ::Integer, idx...) where {S}
	nz = trailing_zeros(S)
	mask = IntSet(1) << nz
	# @info "$S skip $mask"
	_getindex_spaces(Val{S ⊻ mask}, idx...)
end

function _getindex_spaces(::Type{Val{S}}, notint, idx...) where {S}
	nz = trailing_zeros(S)
	mask = IntSet(1) << nz
 	# @info "$S keep $mask"
	return mask | _getindex_spaces(Val{S ⊻ mask}, idx...)
end

@inline function _getindex_dims(S, dims, keep, off1 = 0, off2 = 0)
	i = 1
	j = 0
	nz = trailing_zeros(S)
	cumdims = cumsum(keep)
	dims_ = MVector{length(keep), Int}(undef)
	@inbounds for i in eachindex(keep)
		if keep[i]
			j += 1
			dims_[j] = cumdims[dims[i] - off1] + off2
		end
		S = xor(S, 1 << nz)
		nz = trailing_zeros(S)
	end
	return dims_
end


#-----------------------

reshape(M::MultiMatrix, shape::Dims) = MultiMatrix(reshape(M.data, shape), lspaces(M), rspaces(M))
permutedims(M::MultiMatrix, ord) = MultiMatrix(permutedims(M.data, ord), M)


# TODO: Make a lazy wrapper, just like Base does.
# Note, "partial adjoint" doesn't really make sense.
function adjoint(M::MultiMatrix{LS,RS}) where {LS,RS}
	NL = nlspaces(M)
	NR = nrspaces(M)
	perm = ntuple(i -> i <= NR ? NL+i : i-NL, Val{NL+NR})
	# adjoint is called element-by-element (i.e. it recurses as required)
	return MultiMatrix{RS,LS}(permutedims(adjoint.(M.data), perm), M.rdims .- NL, M.ldims .+ NL)
end


#Swap left and right dimensions of a MultiMatrix
function transpose(M::MultiMatrix{LS,RS}) where {LS,RS}
	NL = nlspaces(M)
	NR = nrspaces(M)
	#perm = ntuple(i -> i <= NR ? NL+i : i-NL, Val{NL+NR})
	perm = (tupseq(NL+1, NL+NR)..., oneto(Val{NL})...)
	return MultiMatrix{RS,LS}(permutedims(M.data, perm), M.rdims .- NR, M.ldims .+ NR)
end


##------------------------
# The following functions take spaces (not dimensions) as arguments.

# Partial transpose
transpose(M::MultiMatrix, ts::Int) = transpose(M, Val{(ts,)})
transpose(M::MultiMatrix, ts::Dims) = transpose(M, Val{ts})
function transpose(M::MultiMatrix{LS,RS}, ::Type{Val{tspaces}}) where {LS, RS, tspaces}
	# This is harder with possibly different left and right spaces

	# NL = nlspaces(M)
	# NR = nrspaces(M)

	TS = binteger(IntSet, Val{tspaces})
	TSL = TS & LS		# transposed spaces on the left side
	TSR = TS & RS		# transposed spaces on the right side


	if iszero(TSL) && iszero(TSR)
		# nothing to transpose
		return M
	elseif TSL == TS && TSR == TS
		# All the spaces are in both left and right.
		# We just need to permute the ltrans and rtrans
		ltrans = findnzbits(TS, LS)		# indices of left spaces to be transposed
		rtrans = findnzbits(TS, RS)		# indices of right spaces to be transposed
		ltdims = M.ldims[ltrans]
		rtdims = M.rdims[rtrans]
		lperm = setindex(M.ldims, rtdims, ltrans)
		rperm = setindex(M.rdims, ltdims, rtrans)
		# @info lperm, rperm
		arr = permutedims(M.data, (lperm..., rperm...))
		return MultiMatrix{LS,RS}(arr, M.ldims, M.rdims)
	else
		ltrans = findnzbits(TS, LS)		# indices of left spaces to be transposed
		rtrans = findnzbits(TS, RS)		# indices of right spaces to be transposed
		lkeep = findnzbits(Val{~TS}, Val{LS})	# left spaces to keep as left
		rkeep = findnzbits(Val{~TS}, Val{RS})	# right spaces to keep as right

		# new left and right spaces
		LS_ = (LS & ~TS) | TSR
		RS_ = (RS & ~TS) | TSL

		# @info "new left spaces = $(findnzbits(LS_)), new right spaces = $(findnzbits(RS_))"

		perm = (M.ldims[lkeep]..., M.rdims[rtrans]..., M.rdims[rkeep]..., M.ldims[ltrans]...)

		lnew = findnzbits(TSR & LS_)		# indices in LS_ of the swapped dimensions
		lkept = findnzbits(~TSR & LS_)	# indices in LS_ of the kept dimensions

		rnew = findnzbits(TSL & RS_)		# indices in RS_ of the swapped dimensions
		rkept = findnzbits(~TSL & RS_)	# indices in RS_ of the kept dimensions

		ldims_ = invpermute((lkept..., lnew...), oneto(Val{count_ones(LS_)}))
		rdims_ = invpermute((rkept..., rnew...), oneto(Val{count_ones(RS_)}))

		arr = permutedims(M.data, perm)
		return MultiMatrix{LS_,RS_}(arr, ldims_, rdims_)
		# @info typeof(M_)
		# @info findnzbits(lspace_int(M_)), findnzbits(rspace_int(M_))
	end
end



# Attempt to make the compiler infer the permutation order
# function transpose2(M::MultiMatrix, ts::Dims)
# 	S = spaces(M)
# 	order = get_transpose_order(Val{S}, Val{ts})
# 	return MultiMatrix(permutedims(M.data, order), S; checkspaces = false)
# end
#
# @generated function get_transpose_order(::Type{Val{S}}, ::Type{Val{ts}}) where {S,ts}
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

macro diagonal_op(M, Idx, Op)
	escM = esc(M)
	escIdx = esc(Idx)
	escOp = esc(Op)
	quote
		if $escM.ldims == $escM.rdims .- nlspaces($escM)
			# Perfectly square -- no permuting necessary
			# sum along diagonal
			n = prod(lsize($escM))
			@inbounds for $escIdx in 1:n+1:n^2
				$escOp
			end
		else
			# Same spaces but in a different order. Need to permute.
			# An explicit loop is faster than reshaping into a matrix
			strides_ = calc_strides(axes($escM))
			strides = strides_[$escM.ldims] .+ strides_[$escM.rdims]
			@inbounds for ci in CartesianIndices(axes($escM, $escM.ldims))
				$escIdx = calc_index(ci, strides)
				$escOp
			end
		end
	end
end


"""
	tr(A)

Returns the trace of a square `MultiMatrix` `A: it contracts each left
space with the corresponding right space and returns a scalar.

	tr(A, spaces = s)

Trace over the the indicated spaces, returning another `MultiMatrix`.
"""
function tr(M::MultiMatrix{LS,RS}) where {LS,RS}
	square(M)	# this is slower
	if LS == RS
		s = zero(eltype(M))
		@diagonal_op M iA s += M.data[iA]
		return s
	else
		error("To trace a MultiMatrix, the left and right spaces must be the same (up to ordering).")
	end
	# return tr(Matrix(square(M)))		# Old way -- much slower
end


# Partial trace.  It's not convenient to use a keyword because it clobbers the method
# that doesn't have a spaces argument.
tr(M::MultiMatrix, space::Integer) = tr(M, (space,))
tr(M::MultiMatrix, spaces::Dims) = tr(M, Val{spaces})

function tr(M::MultiMatrix{LS,RS}, ::Type{Val{tspaces}}) where {LS,RS,tspaces}
	# Int representation of traced spaces
	TS = binteger(IntSet, Val{tspaces})

	# Int representation of kept spaces
	KLS = ~TS & LS
	KRS = ~TS & RS

	if iszero(TS & LS) && iszero(TS & RS)
		# nothing to trace
		return M
	elseif iszero(KLS) && iszero(KRS)
		# trace everything
		return tr(M)
	end

	# find dims to be traced and ktp

	ltdims = M.ldims[findnzbits(TS, LS)]
	rtdims = M.rdims[findnzbits(TS, RS)]

	lkdims = M.ldims[findnzbits(KLS, LS)]
	rkdims = M.rdims[findnzbits(KRS, RS)]

	NL = length(lkdims)
	NR = length(rkdims)
	N = NL + NR
	sz = size(M.data)
	R = similar(M.data, (sz[lkdims]..., sz[rkdims]...))
	trace!(1, M.data, :N, 0, R, lkdims, rkdims, ltdims, rtdims)
	return MultiMatrix{KLS, KRS}(R, oneto(NL), tupseq(NL+1, NL+NR))
end


#
# Operations with UniformScaling
#

# Can only add or subtract I if the MultiMatrix is square
function +(M::MultiMatrix, II::UniformScaling)
	square(M)
	R = LinearAlgebra.copy_oftype(M.data, promote_op(+, eltype(II), eltype(M)))
	@diagonal_op M iR R[iR] += II.λ
	return MultiMatrix(R, M)
end
(+)(II::UniformScaling, M::MultiMatrix) = M + II

function -(M::MultiMatrix, II::UniformScaling)
	square(M)
	R = LinearAlgebra.copy_oftype(M.data, promote_op(-, eltype(M), eltype(II)))
	@diagonal_op M iR R[iR] -= II.λ
	return MultiMatrix(R, M)
end

# NOTE:  This assumes x - A = x + (-A).
function -(II::UniformScaling, M::MultiMatrix)
	square(M)
	R = LinearAlgebra.copy_oftype(-M.data, promote_op(-, eltype(II), eltype(M)))
	@diagonal_op M iR R[iR] += II.λ
	return MultiMatrix(R, M)
end

*(M::MultiMatrix, II::UniformScaling) = M * II.λ
*(II::UniformScaling, M::MultiMatrix) = II.λ * M



#
# Arithmetic operations
#
-(M::MultiMatrix) = MultiMatrix(-M.data, M)

# Fallback methods (when x is not an abstract array or number)
*(M::MultiMatrix, x) = MultiMatrix(M.data * x, M)
*(x, M::MultiMatrix) = MultiMatrix(x * M.data, M)
/(M::MultiMatrix, x) = MultiMatrix(M.data / x, M)

*(M::MultiMatrix, x::Number) = MultiMatrix(M.data * x, M)
*(x::Number, M::MultiMatrix) = MultiMatrix(x * M.data, M)
/(M::MultiMatrix, x::Number) = MultiMatrix(M.data / x, M)

# TODO:  Handle M .+ x and M .- x so that it returns a MultiMatrix

# TODO - HERE IS WHERE I AM WORKING.  CAN IT BE FASTER?
function +(A::MultiMatrix{LS,RS}, B::MultiMatrix{LS,RS}) where {LS,RS}
	if A.ldims == B.ldims && A.rdims == B.rdims
		# Same spaces in the same order
		return MultiMatrix(A.data + B.data, A)
	else
		# Same spaces in different order
		NL = nlspaces(A)
		NR = nrspaces(A)

		# Same code as in ==. Should we make a macro?
		Adims = (A.ldims..., A.rdims...)
		Bdims = (B.ldims..., B.rdims...)

		BinA = Adims[invperm(Bdims)]

		Rdata = A.data .+ permutedims(B, BinA)
		return MultiMatrix(Rdata, A)
		# Rdata = copy(A.data)
		# tensoradd!(1, B.data, Bdims, 1, Rdata, Adims)

		# This is slower
		# Astrides = calc_strides(axes(A))[Adims]
		# Bstrides = calc_strides(axes(B))[Bdims]
		#
		# axR = axes(A)[Adims]
		# Rtype = promote_type(arraytype(A), arraytype(B))
		# szR = map(a->last(a)-first(a)+1, axR)
		# # szR = ntuple(i -> last(axR[i])-first(axR[i])+1, Val(ndims(A)))
		# Rdata = Rtype(undef, szR)
		#
		# @inbounds for ci in CartesianIndices(axR)
		# 	iA = calc_index(ci, Astrides)
		# 	iB = calc_index(ci, Bstrides)
		# 	Rdata[ci] = A[iA] + B[iB]
		# end

		# TODO: Define a simplified constructor for this?
		#			Would using Val() in tupseq help?
		# return MultiMatrix{LS,RS}(Rdata, oneto(NL), tupseq(NL+1, NL+NR))
	end
end


function -(A::MultiMatrix, B::MultiMatrix)
	spaces(A) == spaces(B) || error("To subtract MultiMatrices, they must have the same spaces in the same order")
	axes(A) == axes(B) || throw(DimensionMismatch("To subtract MultiMatrices, they must have the same axes; got axes(A) = $(axes(A)), axes(B) = $(axes(B))"))
	return MultiMatrix(A.data - B.data, Val{lspaces(A)}, Val{rspaces(A)})
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

function _mult_MA(M::MultiMatrix, A::AbstractArray)
	nl = nlspaces(M)
	nr = nrspaces(M)
	S = rspaces(M)
	raxes(M) == axes(A, S) || throw(DimensionMismatch("raxes(M) must equal axes(B, rspaces(A))"))

	nR = max(ndims(A), maximum(S))
	odimsA = deleteat(oneto(nR), S)

	szR = MVector(size(A, oneto(nR)))
	lszM = lsize(M)
	for i = 1:nl
		szR[S[i]] = lszM[i]
	end
	TR = promote_op(*, eltype(M), eltype(A))
	R = similar(A, TR, Tuple(szR))
	contract!(one(eltype(M)), M.data, :N, A, :N, zero(TR), R, oneto(nl), tupseq(nl+1,nl+nr), odimsA, S, invperm((S...,odimsA...)))
	return R
end
#*(M::MultiMatrix, A::AbstractArray{TA,1}) where {TA} =

*(A::AbstractArray{TA}, M::MultiMatrix) where {TA} = _mult_AM(A, M)
*(A::AbstractArray{TA,1}, M::MultiMatrix) where {TA} = _mult_AM(A, M)
*(A::AbstractArray{TA,2}, M::MultiMatrix) where {TA} = _mult_AM(A, M)

# This is almost identical to the M*A version.
function _mult_AM(A::AbstractArray, M::MultiMatrix)
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
	TR = promote_op(*, eltype(M), eltype(A))
	R = similar(A, TR, Tuple(szR))
	contract!(one(eltype(M)), M.data, :N, A, :N, zero(TR), R, tupseq(n+1,2*n), oneto(n), kdimsA, S, invperm((S...,kdimsA...)))
	return R
end


"""
`A*B` where `A` and `B` are MultiMatrices.
"""
function *(A::MultiMatrix, B::MultiMatrix)
	#Atype = arraytype(A).name.wrapper
	#Btype = arraytype(B).name.wrapper
	#Atype == Btype || error("To multiply MultiMatrices, the underlying array types must be the same.  Had types $Atype and $Btype")
	lszA = lsize(A)
	rszA = rsize(A)
	lszB = lsize(B)
	rszB = rsize(B)

	#nA = nspaces(A)
	#nB = nspaces(B)
	rSA = rspaces(A)
	lSB = lspaces(B)

	if rSA == lSB
		# Simple case:  rspaces(A) = lspaces(B)
		raxes(A) == laxes(B) || throw(DimensionMismatch("raxes(A) = $(raxes(A)) must equal laxes(B) = $(laxes(B))"))
		R = reshape(Matrix(A) * Matrix(B), tcat(lspaces(A), rspaces(B)))
		return MultiMatrix(R, Val{lspaces(A)}, Val{rspaces(B)})
	else
		# General case
		(tdimsA, tdimsB, odimsA, odimsB, ldimsR, rdimsR, LSR, RSR) = get_mult_dims(Val{lspaces(A)}, Val{rspaces(A)}, Val{lspaces(B)}, Val{rspaces(B)})
		# println("tdims A,B = $tdimsA <--> $tdimsB")
		# println("odimsA = $odimsA")
		# println("odimsB = $odimsB")
		# println("dimsR = $dimsR")
		# println("spacesR = $SR")

		axes(A, tdimsA) == axes(B, tdimsB) || throw(DimensionMismatch("raxes(A) must equal laxes(B) on spaces common to A,B"))

		szAB = tcat(size(A, odimsA), size(B, odimsB))
		szR = szAB[tcat(ldimsR, rdimsR)]

		TR = promote_op(*, eltype(A), eltype(B))
		R = Array{TR}(undef, szR)
		# println("tdimsA = ", tdimsA)
		# println("odimsA = ", odimsA)
		# println("tdimsB = ", tdimsB)
		# println("odimsB = ", odimsB)
		# println("ldimsR = ", ldimsR)
		# println("rdimsR = ", rdimsR)
		# println("LSR, RSR = ", LSR, RSR)
		contract!(one(eltype(A)), A.data, :N, B.data, :N, zero(TR), R, odimsA, tdimsA, odimsB, tdimsB, ldimsR, rdimsR, nothing)
		return MultiMatrix(R, Val{LSR}, Val{RSR})
		#return nothing
	end
end


# Given lspaces(A), rspaces(A), lspaces(B), rspaces(B), determine the indices for contraction

@generated function get_mult_dims(::Type{Val{LSA}}, ::Type{Val{RSA}}, ::Type{Val{LSB}}, ::Type{Val{RSB}}) where {LSA,RSA,LSB,RSB}
	# Example:
	#    C[o1,o2,o3,o4; o1_,o2_,o3_] = A[o1, o2, o3; o1_, c1, c2] * B[c1, o4, c2; o2_, o3_]
	# itsa, iosa = indices of contracted, open) right spaces of A = (2,3), (1,)
	# itsb, iosb = indicices of contracted, open left spaces of B = (1,3), (2,)
	
	# odimsA = 1:nA, nA+iosa
	# tdimsA = nA + itsa
	# tdimsB = itsb
	# odimsB = iosb, nB+(1:nB)

	# find the dimensions of A,B to be contracted
	nlA = length(LSA)
	nrA = length(RSA)
	nlB = length(LSB)
	nrB = length(RSB)
	ita_ = MVector{nrA,Int}(undef)			# vector of right dims of A to be traced
	itb_ = MVector{nlB,Int}(undef)			# vector of left dims of B to be traces
	oa_mask = @MVector ones(Bool, nrA)		# mask for open right dims of A
	ob_mask = @MVector ones(Bool, nlB)		# maks for open left dims of B

	nt = 0
	for i = 1:nrA
		for j = 1:nlB
	 		if RSA[i] == LSB[j]
				nt += 1
	 			ita_[nt] = i
	 			itb_[nt] = j
				oa_mask[i] = false
				ob_mask[j] = false
	 		end
	 	end
	end

	# indices into RSA and LSB of traced dimensions 
	ita = Tuple(ita_)[oneto(nt)]
	itb = Tuple(itb_)[oneto(nt)]
	# dimensions to be traced
	tdimsA = nlA .+ ita
	tdimsB = itb

	# indices into RSA and LSB of open dimensions
	iora = oneto(nrA)[oa_mask]
	iolb = oneto(nlB)[ob_mask]
	
	norA = nrA - nt		# number of open spaces of A
	nolB = nlB - nt		# number of open spaces of B

	odimsA = tcat(oneto(nlA), nlA .+ iora)
	odimsB = tcat(iolb, tupseq(nlB+1, nlB+nrB))


	# dimensions of output R
	LSR = tcat(LSA, LSB[iolb])
	RSR = tcat(RSA[iora], RSB)

	nlR = nlA + nolB
	nrR = norA + nrB

	nodA = length(odimsA)
	# indices into tcat(odimsA, odimsB)
	ldimsR = tcat(oneto(nlA), tupseq(nodA + 1, nodA + nolB))
	rdimsR = tcat(tupseq(nlA + 1, nlA + norA), tupseq(nodA + nolB + 1, nodA + nolB + nrB)) 
	#dimsR = tcat(itsa, iosa, nodA .+ oneto(nosB), (nodA + nosB) .+ itsb, tupseq(nA+1, nA+nosA), (nodA + nosB) .+ iosb)
	return :( ($tdimsA, $tdimsB, $odimsA, $odimsB, $ldimsR, $rdimsR, $LSR, $RSR) )
end


^(M::MultiMatrix, x::Number) = MultiMatrix(reshape(Matrix(M)^x, size(M)), Val{spaces(M)}; checkspaces = false)
^(M::MultiMatrix, x::Integer) = MultiMatrix(reshape(Matrix(M)^x, size(M)), Val{spaces(M)}; checkspaces = false)

#
# Analytic matrix functions
#

for f in [:inv, :sqrt, :exp, :log, :sin, :cos, :tan, :sinh, :cosh, :tanh]
	@eval function $f(M::MultiMatrix)
			chk_square(M)
			MultiMatrix(reshape($f(Matrix(M)), size(M)), Val{spaces(M)}; checkspaces = false)
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
MultiMatrixStyle{S,A,N}(::Type{Val{N}}) where {S,A,N} = MultiMatrixStyle{S,A,N}()

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
