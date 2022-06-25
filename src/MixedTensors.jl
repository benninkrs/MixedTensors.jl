"""
	MixedTensors (module)

Implements arbitrary tensors acting on labelled vector spaces and/or their duals. 
"""


#=
Function that take dimensions as inputs:
	size, axes, pernmutedims

Functions that take spaces as inputs:
	transpose
=#


#=
** Currently working on getindex, working my way down the file ** 
TODO: getindex is still kinda slow (specifically the part M.data[idx...])
TODO: lperm/rperm instead of ldims/rdims?
TODO: reinstitute bypassing checking on the constructor
TODO: Use dispatch to separate out A*B with same spaces
TODO: Extend addition, subtraction to allow spaces to be in different order
TODO: Extend * to form lazy outer products
TODO: Support +,- for dissimilar spaces?  Perhaps lazy structures?
TODO: Figure out why broadcasting is slow (slower than map/ntuple, at least)
TODO: Use Strided.jl?
TODO: Support in-place operations?
TODO: Check validity of dims in lsize, rsize, laxes, raxes
TODO: generalize + to >2 arguments
TODO: Better support of different underlying array types?
			In places (e.g. *) it is assumed that the underlying array type has
			the element type as the first parameter and has a constructor of the
			form ArrayType{ElementType}(undef, size).
=#

module MixedTensors

export Tensor, lsize, rsize, spaces, lspaces, rspaces, nlspaces, nrspaces
export TestStruct, test_slurp, test_tuple

using MiscUtils
using SuperTuples
using StaticArrays
using LinearAlgebra
using TensorOperations: trace!, contract!, tensoradd!
using Base.Broadcast: Broadcasted, BroadcastStyle
using Base: tail

import Base: display, show
import Base: ndims, length, size, axes, similar
import Base: reshape, permutedims, adjoint, transpose, Matrix, ==
import Base: getindex, setindex!
import Base: (+), (-), (*), (/), (^)
import Base: inv, exp, log, sin, cos, tan, sinh, cosh, tanh

import LinearAlgebra: tr, eigvals, svdvals, opnorm
import Base: similar
import SuperTuples.static_fn



#------------------------------------
# Preliminaries & utilities


const SpacesInt = UInt128  		# An integer treated as a bit set for vector spaces
const Iterable = Union{Tuple, AbstractArray, UnitRange, Base.Generator}
const Axes{N} = NTuple{N, AbstractUnitRange{<:Integer}}

function maskbits(::Val{I}, ::Val{mask}) where {I} where {mask}
	#isa(mask, NTuple{Bool}) || error("mask must be a NTuple{Bool}")
	I isa SpacesInt || error("I must be a SpacesInt")
	length(mask) == count_ones(I) || error("must have length(mask) == count_ones(I)")

	maskbits_(Val(I), Val(mask))
end


function maskbits_(::Val{I}, ::Val{mask}) where {I} where {mask}
	# print("  I = ", I, "  mask = ", mask)
	N = length(mask)
	if N == 0 
		# println("")
		return SpacesInt(0)
	else
		i = trailing_zeros(I) + 1
		I_ = (I >> i) << i 
		J = mask[1] ? SpacesInt(1) << (i-1) : SpacesInt(0)
		# println("  i = ", i, "  bitmask = ", J)
		return maskbits_(Val(I_), Val(tail(mask))) | J
	end
end

# function maskbits_(::Val{I}, ::Val{mask}, J::SpacesInt) where {I,mask}
# 	print("  I = ", I, "  mask = ", mask)
# if length(mask) == 0
# 		return SpacesInt(0)
# 	elseif mask[1]
# 		i = trailing_zeros(I)
# 		J = J << i
# 		println("  i = ", i, "  bitmask = ", J)

# 		return maskbits_(Val(I>>i), Val(tail(mask)), J) | J
# 	else
# 		return maskbits_(Val(I>>i), Val(tail(mask)), J)
# 	end
# end 

# function maskbits(::Val{I}, ::Val{mask}) where {I} where {mask}
# 	#isa(mask, NTuple{Bool}) || error("mask must be a NTuple{Bool}")
# 	N = length(mask)
# 	I isa SpacesInt || error("I must be a SpacesInt")
# 	#print("N = ", N, "  I = ", I, "  mask = ", mask)

# 	if N == 0 
# 		return I
# 	else
# 		i = trailing_zeros(I) + 1
# 		I_ = (I >> i) << i 
# 		#println("  i = ", i, "  bitmask = ", SpacesInt(mask[1]) << (i-1))
# 		return  (SpacesInt(mask[1]) << (i-1)) | maskbits(Val(I_), Val(tail(mask)))
# 	end
# end


# Compute the dimensions of selected spaces after the others are contracted out
function remain_dims(dims::Dims{N}, ::Val{S}, ::Val{K}) where {N, S, K}
	# S and K should be SpaceInts
	count_ones(S) == N || error("count_ones(S) must equal length(dims)")
	idims = findnzbits(K, S)
	kdims = dims[idims]
	dims_ = sortperm(kdims)		# it infers!
	# Should be equivalent
	# dims_ = ntuple(Val(N_)) do i
	# 	static_fn(0, Val(length(kdims))) do a,j
	# 		kdims[j] <= kdims[i] ? a+1 : a		
	# 	end
	# end
end



# Can we find a better approach to calc_strides and diagonal_op?
calc_strides(sz::Dims{N}) where {N} = cumprod(ntuple(i -> i==1 ? 1 : sz[i-1], Val(N)))
calc_strides(ax::Axes{N}) where {N} = cumprod(ntuple(i -> i==1 ? 1 : last(ax[i-1]) - first(ax[i-1]) + 1, Val(N)))
@inline calc_index(I::CartesianIndex{N}, strides::NTuple{N,Int}) where {N} = 1 + sum((Tuple(I) .- 1) .* strides)

# Perform an operation along the diagonal of an array
# PROBABLY DOESN'T WORK FOR NON-SQUARE TENSORS
# Also, are some of my more recent approaches (a la DiagonalIndexing) faster?
macro diagonal_op(M, Idx, Op)
	# M is the Tensor, Idx is the index, Op is the operation
	escM = esc(M)
	escIdx = esc(Idx)
	escOp = esc(Op)
	quote
		@warn "diagonal_op may not be correct for tensors with different left and right spaces"
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



#--------------------------------
# Constructors

"""
A (m,n) `Tensor` is a multidimensional array of m+n dimensions; the first m dimensions
are associated with a set of "left" vector spaces (l1,...,lm) and the last n dimensions
are associated with a set of "right" vector spaces (r1,...,rn). Each space is
designated by an integer ∈ 1,...,128. Left and right spaces with the same index are adjoint (dual).
"""
struct Tensor{LS, RS, T, N, A<:DenseArray{T,N}, NL, NR} <: DenseArray{T,N}
	# LS and RS are wide unsigned integers whose bits indicate the left and right spaces
	# ldims and rdimes map the left and right spaces to corresponding dimensions of A.
	# (all the left dimensions precede all the right dimensions)
	data::A
	ldims::Dims{NL}		# dims of A corresponding to the ordered left spaces
	rdims::Dims{NR}		# dims of A corresponding to the ordered right spaces

	# Primary inner constructor. Validates inputs
	function Tensor{LS, RS}(data::A, ldims::NTuple{NL,Integer}, rdims::NTuple{NR,Integer}) where {LS, RS, NL, NR} where {A<:DenseArray{T,N}} where {T,N}
		count_ones(LS) == NL || error("length(ldims) must equal count_ones(LS)")
		count_ones(RS) == NR || error("length(rdims) must equal count_ones(RS)")
		# isperm((ldims...,rdims...)) || error("ldims and rdims must form a permutation of the array's dimensions")
		NL + NR == N || error("ndims(A) must be equal the total number of spaces (left + right)")
		return new{LS, RS, T, N, A, NL, NR}(data, ldims, rdims)
	end

	# Construct from non-DenseArrays
	Tensor{LS,RS}(data::AbstractArray, ldims, rdims) where {LS,RS} = Tensor{LS,RS}(collect(data), ldims, rdims)

	# Shortcut constructor: construct from array, using another Tensor's metadata.
	# By ensuring the array has the right number of dimensions, no input checking is needed.
	function Tensor(arr::A_, M::Tensor{LS,RS,T,N,A,NL,NR}) where {A_ <: DenseArray{T_,N}} where {T_} where {LS,RS,T,N,A,NL,NR}
		return new{LS,RS,T_,N,A_,NL,NR}(arr, M.ldims, M.rdims)
	end

end

struct TestStruct{A<:DenseArray}
	data::A
end

# Convenience constructors

"""
	Tensor(A::AbstractArray, lspaces::Tuple, rspaces::Tuple)

Create from array 'A' a Tensor that acts on specified vector spaces.
The first `length(lspaces)` dimensions of `A` are assumed to correspond the left spaces.
The last `length(rspaces)` dimensions of `A` are assumed to correspond to the right spaces.

	Tensor(A::AbstractArray, spaces::Tuple)

creaes a tensor with identical left and right spaces.

	Tensor(A::AbstractArray)

creates a left vector with `lspaces = 1:ndims(A)` and `rspaces = ()`.

"""
Tensor(arr) = Tensor(arr, Val(oneto(ndims(arr))), Val(()))

Tensor(arr, spaces) = Tensor(arr, tuple(spaces...))
Tensor(arr, spaces::Dims) = Tensor(arr, Val(spaces))

Tensor(arr, lspaces, rspaces) = Tensor(arr, tuple(lspaces...), tuple(rspaces...))
Tensor(arr, lspaces::Dims, rspaces::Dims) = Tensor(arr, Val(lspaces), Val(rspaces))


# internal constructors
function Tensor(arr::A, ::Val{spaces}) where {spaces} where A<:AbstractArray{T,N} where {T,N}
	lorder = sortperm(spaces)
	IS = binteger(SpacesInt, Val(spaces))
	Tensor{IS,IS}(arr, lorder, lorder .+ length(spaces))
end

function Tensor(arr::A, ::Val{Lspaces}, ::Val{Rspaces}) where {Lspaces, Rspaces} where A<:AbstractArray{T,N} where {T,N}
	lorder = sortperm(Lspaces)
	rorder = sortperm(Rspaces)
	IL = binteger(SpacesInt, Val(Lspaces))
	IR = binteger(SpacesInt, Val(Rspaces))
	Tensor{IL,IR}(arr, lorder, rorder .+ length(Lspaces))
end



# Reconstruct Tensor with different spaces
"""
	(T::Tensor)(spaces...)
	(T::Tensor)(spaces::Tuple)
	(T::Tensor)(lspaces::Tuple, rspaces::Tuple)`

Create a Tensor with the same data as `M` but acting on different spaces.
(This is a lazy operation that is generally to be preferred over [`permutedims`](@ref).)
"""
(M::Tensor)(spaces::Vararg{Int64}) = M(spaces)						# for list of ints
(M::Tensor)(spaces) = Tensor(M.data, tuple(spaces...))			# for iterable
(M::Tensor)(spaces::Dims) = Tensor(M.data, Val(spaces))
(M::Tensor)(::Val{S}) where {S} = Tensor(M.data, Val(S))

(M::Tensor)(lspaces, rspaces) where {N} = Tensor(M.data, tuple(lspaces...), tuple(rspaces...))
(M::Tensor)(lspaces::Dims, rspaces::Dims) where {N} = Tensor(M.data, Val(lspaces), Val(rspaces))
(M::Tensor)(::Val{LS}, ::Val{RS}) where {LS,RS} = Tensor(M.data, Val(LS), Val(RS))


# Reconstruct with different spaces
function Tensor(M::Tensor{IL,IR}, ::Val{Lspaces}, ::Val{Rspaces}) where {IL,IR} where {Lspaces, Rspaces} where A<:AbstractArray{T,N} where {T,N}
	IL_ = binteger(SpacesInt, Val(Lspaces))
	IR_ = binteger(SpacesInt, Val(Rspaces))
	lperm = sortperm(Lspaces)
	rperm = sortperm(Rspaces)
	Tensor{IL_,IR_}(M.data, M.ldims[lperm], M.rdims[rperm])
end


# Similar array with the same spaces and same size

# similar(M::Tensor{LS,RS,T,N,A}) where {LS,RS,T,N,A} = Tensor{LS,RS,T,N,A}(similar(M.data))
# # Similar array with different type, but same size and spaces.
# similar(M::Tensor{LS,RS,T,N}, newT) = Tensor{LS,RS,newT,N}(similar(M.data, newT), M.lperm, M.rperm)
#
# #Similar array with different type and/or size
# const Shape = Tuple{Union{Integer, Base.OneTo},Vararg{Union{Integer, Base.OneTo}}}
#
# similar(::Type{M}) where {M<:Tensor{LS,RS,T,N,A}} where {LS,RS} where {A<:DenseArray{T,N}} where {T,N} = Tensor{LS,RS,T,N,A}(similar(A, shape))

#-------------
# Size and shape
ndims(M::Tensor) = ndims(M.data)


lspaces_int(M::Tensor{LS, RS}) where {LS,RS} = LS
rspaces_int(M::Tensor{LS, RS}) where {LS,RS} = RS

nlspaces(M::Tensor{LS, RS, T, N, A, NL, NR}) where {LS, RS, T, N, A, NL, NR} = NL
nrspaces(M::Tensor{LS, RS, T, N, A, NL, NR}) where {LS, RS, T, N, A, NL, NR} = NR

# Return the spaces in array order
spaces(M::Tensor) = (lspaces(M), rspaces(M))
lspaces(M::Tensor{LS}) where {LS} = invpermute(findnzbits(Val(LS)), M.ldims)
rspaces(M::Tensor{LS, RS, T, N, A, NL, NR}) where {LS, RS, T, N, A, NL, NR} = invpermute(findnzbits(RS), M.rdims .- NL)


length(M::Tensor) = length(M.data)

size(M::Tensor) = size(M.data)
size(M::Tensor, dim) = size(M.data, dim)
size(M::Tensor, dims::Iterable) = size(M.data, dims)

axes(M::Tensor) = axes(M.data)
axes(M::Tensor, dim) = axes(M.data, dim)
axes(M::Tensor, dims::Iterable) = map(d->axes(M.data, d), dims)

lsize(M::Tensor) = ntuple(i -> size(M.data, M.ldims[i]), Val(nlspaces(M)))
rsize(M::Tensor) = ntuple(i -> size(M.data, M.rdims[i]), Val(nrspaces(M)))

# Not sure laxes and raxes are really necessary
laxes(M::Tensor) = ntuple(i -> axes(M.data, M.ldims[i]), Val(nlspaces(M)))
raxes(M::Tensor) = ntuple(i -> axes(M.data, M.rdims[i]), Val(nrspaces(M)))


arraytype(::Tensor{LS,RS,T,N,A} where {LS,RS,T,N}) where A = A


size2string(d) = isempty(d) ? "0-dim" :
                 length(d) == 1 ? "length-$(d[1])" :
                 join(map(string,d), '×')

display(M::Tensor) = show(M)

function show(io::IO, M::Tensor)
	# print as usual
	print(io, size2string(size(M.data)), " ")
	print(io, "Tensor{", lspaces(M), "←→", rspaces(M), ", ", arraytype(M), "}")

	if !isempty(M.data)
		println(io, ':')
		Base.print_array(IOContext(io, :compact => true, :typeinfo => eltype(M.data)), M.data)
	end
end


# # Return the type of an array
# function promote_arraytype(A::Tensor, B::Tensor)
# 	# This only works if the array type has the dimension as the last parameter.
# 	# Uh oh ... how could even know where the dimension parameter is?
#   return freelastparameter(promote_type(arraytype(A), arraytype(B)))
# end


# X and Y have the same spaces
function ==(X::Tensor{LS,RS}, Y::Tensor{LS,RS}) where {LS, RS}
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

	# It is probably possible to do this even faster using a macro
	Xstrides = calc_strides(axes(X))[Xdims]
	Ystrides = calc_strides(axes(Y))[Ydims]

	for ci in CartesianIndices(ax)
		iX = calc_index(ci, Xstrides)	 # computing the index each time probably costs more than updating the index based on the strides
		iY = calc_index(ci, Ystrides)
		if X[iX] != Y[iY]
			return false
		end
	end
	return true

end

# Fallback when X,Y have different spaces
==(X::Tensor, Y::Tensor) = false


# Internal functions to make ensure a Tensor is "square".
# A Tensor M is *square* if it has the same left and right spaces (in any order)
# and the corresponding left and right axes are the same.
# It is "proper" square if it is square and the left and right spaces are in the same order.
#
# ensure_square(M) returns the square version of M (either M or a permutation of M) if it exists;
# otherwise throws an error
@inline function ensure_square(M::Tensor{LS,LS}) where {LS}
	axes(M, M.ldims) == axes(M, M.rdims) ? M : throw(DimensionMismatch("Tensor is not square"))
end

# This is the fallback, in the case M's left and right spaces are different
@inline ensure_square(M::Tensor{LS,RS}) where {LS,RS} = throw(DimensionMismatch("Tensor is not square"))




# Conversions
convert(T, M::Tensor) = convert(T, M.data)


"""
`Matrix(M::Tensor)`

Convert `M` to a `Matrix`. The left (right) dimensions of `M` are reshaped
into the first (second) dimension of the output matrix.
"""
Matrix(M::Tensor) = reshape(M.data, (prod(lsize(M)), prod(rsize(M))))



# ------------------------
# Array access

function unindexed_dims(dims::Dims{N}, dkeep, ::Val{S}) where {N, S}
	(S isa SpacesInt) || error("S must be a SpacseInt")
	count_ones(S) == N || error("count_ones(S) must equal length(dims)")

	spaces = findnzbits(Val(S))
	skeep = findin(dims, dkeep)
	spaces_ = spaces[skeep]
	dims_ = sortperm(spaces_)
	S_ = binteger(SpacesInt, spaces_)
	return (S_, dims_)
end

# If accessing a single element, return that element.
"""
	getindex(T::Tensor, idx...)

Indexing a Tensor indexes the backing array.  If all the indices are scalars the result
is the specified array element.  Otherwise the result is a Tensor whose spaces are those
associated with the dimensions that were indexed by non-scalars.
"""
getindex(M::Tensor, i::Vararg{Union{Integer, CartesianIndex}}) = getindex(M.data, i...)

# Index with ranges returns a Tensor.
# Spaces indexed by scalars are removed from the Tensor
getindex(M::Tensor, idx...) = getindex_(M, idx)
function getindex_(M::Tensor{LS,RS}, idx) where {LS,RS}		# for some reason, slurping/Vararg is slow
	# I = to_indices(M.data, idx)
	NL = nlspaces(M)
	NR = nrspaces(M)

	# Tuple of bools indicating which dimensions to keep
	ldmask = ntuple(i -> !isa(idx[i], Integer), Val(NL)) 
	rdmask = ntuple(i -> !isa(idx[i+NL], Integer), Val(NR))

	ldkeep = tfindall(Val(ldmask))
	rdkeep = tfindall(Val(rdmask))
	NL_ = length(ldkeep)
	# NR_ = length(rdkeep)

	(LS_, ldims_) = unindexed_dims(M.ldims, ldkeep, Val(LS))

	rdims = M.rdims .- NL
	(RS_, rdims_) = unindexed_dims(rdims, rdkeep, Val(RS))
	rdims_ = rdims_ .+ NL_
	# rdims_ = map(i -> i+ length(ldims_), rdims_)		#  a little faster

	data_ = M.data[idx...]		# this is slow, much slower than when calling explicity
	Tensor{LS_,RS_}(data_, ldims_, rdims_)
end

#-----------------------

reshape(M::Tensor, shape::Dims) = Tensor(reshape(M.data, shape), lspaces(M), rspaces(M))
permutedims(M::Tensor, ord) = Tensor(permutedims(M.data, ord), M)


# TODO: Make a lazy wrapper, just like Base does.
# Note, "partial adjoint" doesn't really make sense.
function adjoint(M::Tensor{LS,RS}) where {LS,RS}
	NL = nlspaces(M)
	NR = nrspaces(M)
	perm = ntuple(i -> i <= NR ? NL+i : i-NL, Val(NL+NR))
	# adjoint is called element-by-element (i.e. it recurses as required)
	return Tensor{RS,LS}(permutedims(adjoint.(M.data), perm), M.rdims .- NL, M.ldims .+ NL)
end


#Swap left and right dimensions of a Tensor
"""
	transpose(T::Tensor)

Transpose (swap the left and right spaces) of a tensor. 

	transpose(T::Tensor, space)
	transpose(T::Tensor, spaces...)

Transpose selected spaces (partial transpose).  In general the resulting spaces
will be put in sequential order. But if T does not involve the specified spaces,
 `transpose` does nothing.
"""
function transpose(M::Tensor{LS,RS}) where {LS,RS}
	NL = nlspaces(M)
	NR = nrspaces(M)
	#perm = ntuple(i -> i <= NR ? NL+i : i-NL, Val(NL+NR))
	perm = (tupseq(NL+1, NL+NR)..., oneto(Val{NL})...)
	ldims_ = M.rdims .- NL
	rdims_ = M.ldims .+ NR
	# println(perm)
	# println(ldims_)
	# println(rdims_)
	return Tensor{RS,LS}(permutedims(M.data, perm), ldims_, rdims_)
end


# Partial transpose
transpose(M::Tensor, ts::Int) = transpose(M, Val((ts,)))
transpose(M::Tensor, ts::Dims) = transpose(M, Val(ts))
function transpose(M::Tensor{LS,RS}, ::Val{tspaces}) where {LS, RS, tspaces}
	NL = nlspaces(M)
	NR = nrspaces(M)

	TS = binteger(SpacesInt, Val(tspaces))
	TSL = TS & LS		# transposed spaces on the left side
	TSR = TS & RS		# transposed spaces on the right side


	if iszero(TSL) && iszero(TSR)
		# nothing to transpose
		return M
	elseif TSL == TS && TSR == TS
		# All the spaces are in both left and right.
		# We just need to permute the dimensions, the spaces stay the same
		it_lspaces = findnzbits(TS, LS)		# indices of left spaces to be transposed (in space order)
		it_rspaces = findnzbits(TS, RS)		# indices of right spaces to be transposed (in space order)
		tldims = M.ldims[it_lspaces]			# left dimensions to be transposed (in space order)
		trdims = M.rdims[it_rspaces]			# right dimensions to be transposed (in space order)
		lperm = setindex(M.ldims, trdims, it_lspaces)		# the transposed spaces have the same order
		rperm = setindex(M.rdims, tldims, it_rspaces)
		arr = permutedims(M.data, (lperm..., rperm...))
		return Tensor{LS,RS}(arr, oneto(NL), tupseq(NL+1, NL+NR))
		# lperm = setindex(oneto(NL), trdims, tldims)
		# rperm = setindex(tupseq(NL+1,NL+NR), tldims, trdims .- NL)
		# arr = permutedims(M.data, (lperm..., rperm...))
		# return Tensor{LS,RS}(arr, M.ldims, M.rdims)
	else
		it_lspaces = findnzbits(TS, LS)		# indices of left spaces to be transposed
		it_rspaces = findnzbits(TS, RS)		# indices of right spaces to be transposed
		ik_lspaces = findnzbits(~TS, LS)		# indices of left spaces to keep as left
		ik_rspaces = findnzbits(~TS, RS)		# indices of right spaces to keep as right

		# new left and right spaces
		LS_ = (LS & ~TS) | TSR
		RS_ = (RS & ~TS) | TSL
		NL_ = count_ones(LS_)
		NR_ = count_ones(RS_)

		lsp = findnzbits(LS)
		rsp = findnzbits(RS)

		lspaces_ = (lsp[ik_lspaces]..., rsp[it_rspaces]...)
		lperm_ = sortperm(lspaces_)
		ldims_ = (M.ldims[ik_lspaces]..., M.rdims[it_rspaces]...)		# dims of the new left spaces
		lperm = ldims_[lperm_];

		rspaces_ = (rsp[ik_rspaces]..., lsp[it_lspaces]...)
		rperm_ = sortperm(rspaces_)
		rperm = (M.rdims[ik_rspaces]..., M.ldims[it_lspaces]...)[rperm_];

		perm = (lperm..., rperm...)
		
		arr = permutedims(M.data, perm)
		return Tensor{LS_,RS_}(arr, oneto(NL_), tupseq(NL_+1, NL_+NR_))

	end
end




"""
	tr(A)

Returns the trace of a square `Tensor` `A: it contracts each left
space with the corresponding right space and returns a scalar.

	tr(A, spaces = s)

Trace over the the indicated spaces, returning another `Tensor`.
"""
tr(M::Tensor{LS,RS}) where {LS,RS} = error("To trace a Tensor, the left and right spaces must be the same (up to ordering).")
function tr(M::Tensor{LS,LS}) where {LS}
   ensure_square(M)# this is slower
   s = zero(eltype(M))
   @diagonal_op M iA s += M.data[iA]
   return s
end


# Partial trace.  It's not convenient to use a keyword because it clobbers the method
# that doesn't have a spaces argument.
tr(M::Tensor, space::Integer) = tr(M, (space,))
tr(M::Tensor, spaces::Dims) = tr(M, Val(spaces))

function tr(M::Tensor{LS,RS}, ::Val{tspaces}) where {LS,RS,tspaces}
	# Int representation of traced spaces
	TS = binteger(SpacesInt, Val(tspaces))

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
	return Tensor{KLS, KRS}(R, oneto(NL), tupseq(NL+1, NL+NR))
end


#
# Operations with UniformScaling
#

# Can only add or subtract I if the Tensor is square
function +(M::Tensor, II::UniformScaling)
	ensure_square(M)
	R = LinearAlgebra.copy_oftype(M.data, promote_type(eltype(II), eltype(M)))
	@diagonal_op M iR R[iR] += II.λ
	return Tensor(R, M)
end
(+)(II::UniformScaling, M::Tensor) = M + II

function -(M::Tensor, II::UniformScaling)
	ensure_square(M)
	R = LinearAlgebra.copy_oftype(M.data, promote_type(eltype(M), eltype(II)))
	@diagonal_op M iR R[iR] -= II.λ
	return Tensor(R, M)
end

# NOTE:  This assumes x - A = x + (-A).
function -(II::UniformScaling, M::Tensor)
	ensure_square(M)
	R = LinearAlgebra.copy_oftype(-M.data, promote_type(eltype(II), eltype(M)))
	@diagonal_op M iR R[iR] += II.λ
	return Tensor(R, M)
end

*(M::Tensor, II::UniformScaling) = M * II.λ
*(II::UniformScaling, M::Tensor) = II.λ * M



#
# Arithmetic operations
#
-(M::Tensor) = Tensor(-M.data, M)

# Fallback methods (when x is not an abstract array or number)
*(M::Tensor, x) = Tensor(M.data * x, M)
*(x, M::Tensor) = Tensor(x * M.data, M)
/(M::Tensor, x) = Tensor(M.data / x, M)

*(M::Tensor, x::Number) = Tensor(M.data * x, M)
*(x::Number, M::Tensor) = Tensor(x * M.data, M)
/(M::Tensor, x::Number) = Tensor(M.data / x, M)

# TODO:  Handle M .+ x and M .- x so that it returns a Tensor

# TODO - HERE IS WHERE I AM WORKING.  CAN IT BE FASTER?
function +(A::Tensor{LS,RS}, B::Tensor{LS,RS}) where {LS,RS}
	if A.ldims == B.ldims && A.rdims == B.rdims
		# Same spaces in the same order
		return Tensor(A.data + B.data, A)
	else
		# Same spaces in different order
		NL = nlspaces(A)
		NR = nrspaces(A)

		# Same code as in ==. Should we make a macro?
		Adims = (A.ldims..., A.rdims...)
		Bdims = (B.ldims..., B.rdims...)

		BinA = Adims[invperm(Bdims)]

		Rdata = A.data .+ permutedims(B, BinA)
		return Tensor(Rdata, A)
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
		# return Tensor{LS,RS}(Rdata, oneto(NL), tupseq(NL+1, NL+NR))
	end
end


function -(A::Tensor, B::Tensor)
	spaces(A) == spaces(B) || error("To subtract MixedTensors, they must have the same spaces in the same order")
	axes(A) == axes(B) || throw(DimensionMismatch("To subtract MixedTensors, they must have the same axes; got axes(A) = $(axes(A)), axes(B) = $(axes(B))"))
	return Tensor(A.data - B.data, Val(lspaces(A)), Val(rspaces(A)))
end

# Matrix multiplication.  These methods are actually quite fast -- about as fast as the
# core matrix multiplication. We appear to incur very little overhead.
"""
`M*X` where `M` is a Tensor and `X` is an `AbstractArray` contracts the right
dimensions of `M` with dimensions `spaces(M)` of `X`.  The result is an array of type
similar to `X`, whose size along the contracted dimensions is `lsize(M)` and whose size in
the uncontracted dimensions is that of `X`.

`X*M` is similar, except the left dimensions of `M` are contracted against `X`, and the
size of the result depends on the `rsize(M)`.
"""
*(M::Tensor, A::DenseArray{TA}) where {TA} = _mult_MA(M, A)
*(M::Tensor, A::DenseArray{TA,1}) where {TA} = _mult_MA(M, A)
*(M::Tensor, A::DenseArray{TA,2}) where {TA} = _mult_MA(M, A)

function _mult_MA(M::Tensor, A::DenseArray)
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
	TR = promote_type(eltype(M), eltype(A))
	R = similar(A, TR, Tuple(szR))
	contract!(one(eltype(M)), M.data, :N, A, :N, zero(TR), R, oneto(nl), tupseq(nl+1,nl+nr), odimsA, S, invperm((S...,odimsA...)))
	return R
end
#*(M::Tensor, A::DenseArray{TA,1}) where {TA} =

*(A::DenseArray{TA}, M::Tensor) where {TA} = _mult_AM(A, M)
*(A::DenseArray{TA,1}, M::Tensor) where {TA} = _mult_AM(A, M)
*(A::DenseArray{TA,2}, M::Tensor) where {TA} = _mult_AM(A, M)

# This is almost identical to the M*A version.
function _mult_AM(A::DenseArray, M::Tensor)
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
	TR = promote_type(eltype(M), eltype(A))
	R = similar(A, TR, Tuple(szR))
	contract!(one(eltype(M)), M.data, :N, A, :N, zero(TR), R, tupseq(n+1,2*n), oneto(n), kdimsA, S, invperm((S...,kdimsA...)))
	return R
end



"""
`A*B` where `A` and `B` are MixedTensors.
"""
function *(A::Tensor{LSA,RSA}, B::Tensor{LSB,RSB}) where {LSA,RSA} where {LSB,RSB}
	# Check for compatibility of spaces
	SC = RSA & LSB			# spaces to be contracted
	RSA_B = RSA & (~SC)		# right spaces of A, not contracted
	LSB_A = LSB & (~SC)		# left spaces of B, not contracted

	LSA & LSB_A == SpacesInt(0) || error("A and B have uncontracted left spaces in common")
	RSB & RSA_B == SpacesInt(0) || error("A and B have uncontracted right spaces in common")

	LSR = LSA | LSB_A
	RSR = RSB | RSA_B

	error("Function in development")

	#Atype = arraytype(A).name.wrapper
	#Btype = arraytype(B).name.wrapper
	#Atype == Btype || error("To multiply MixedTensors, the underlying array types must be the same.  Had types $Atype and $Btype")
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
		R = reshape(Matrix(A) * Matrix(B), tcat(lsize(A), rsize(B)))
		return Tensor{LSA,RSB}(R, oneto(nlspaces(A)), tupseq(nlspaces(A)+1, nlspaces(A)+nrspaces(B)))
	else
		# General case
		(tdimsA, tdimsB, odimsA, odimsB, ldimsR, rdimsR, LSR, RSR) = get_mult_dims(Val(lspaces(A)), Val(rspaces(A)), Val(lspaces(B)), Val(rspaces(B)))
		# println("tdims A,B = $tdimsA <--> $tdimsB")
		# println("odimsA = $odimsA")
		# println("odimsB = $odimsB")
		# println("dimsR = $dimsR")
		# println("spacesR = $SR")

		axes(A, tdimsA) == axes(B, tdimsB) || throw(DimensionMismatch("raxes(A) must equal laxes(B) on spaces common to A,B"))

		szAB = tcat(size(A, odimsA), size(B, odimsB))
		szR = szAB[tcat(ldimsR, rdimsR)]

		TR = promote_type(eltype(A), eltype(B))
		R = Array{TR}(undef, szR)
		# println("tdimsA = ", tdimsA)
		# println("odimsA = ", odimsA)
		# println("tdimsB = ", tdimsB)
		# println("odimsB = ", odimsB)
		# println("ldimsR = ", ldimsR)
		# println("rdimsR = ", rdimsR)
		# println("LSR, RSR = ", LSR, RSR)
		contract!(one(eltype(A)), A.data, :N, B.data, :N, zero(TR), R, odimsA, tdimsA, odimsB, tdimsB, ldimsR, rdimsR, nothing)
		return Tensor(R, Val(LSR), Val(RSR))
		#return nothing
	end
end


# Given lspaces(A), rspaces(A), lspaces(B), rspaces(B), determine the indices for contraction

@generated function get_mult_dims(::Val{lsa}, ::Val{rsa}, ::Val{lsb}, ::Val{rsb}) where {lsa,rsa,lsb,rsb}
	# Example:
	#    C[o1,o2,o3,o4; o1_,o2_,o3_] = A[o1, o2, o3; o1_, c1, c2] * B[c1, o4, c2; o2_, o3_]
	# itsa, iosa = indices of contracted, open) right spaces of A = (2,3), (1,)
	# itsb, iosb = indicices of contracted, open left spaces of B = (1,3), (2,)
	
	# odimsA = 1:nA, nA+iosa
	# tdimsA = nA + itsa
	# tdimsB = itsb
	# odimsB = iosb, nB+(1:nB)

	# find the dimensions of A,B to be contracted
	# TODO: Would masking help?
	nlA = length(lsa)
	nrA = length(rsa)
	nlB = length(lsb)
	nrB = length(rsb)
	ita_ = MVector{nrA,Int}(undef)			# vector of right dims of A to be traced
	itb_ = MVector{nlB,Int}(undef)			# vector of left dims of B to be traces
	oa_mask = @MVector ones(Bool, nrA)		# mask for open right dims of A
	ob_mask = @MVector ones(Bool, nlB)		# maks for open left dims of B

	nt = 0
	for i = 1:nrA
		for j = 1:nlB
	 		if rsa[i] == lsb[j]
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
	LSR = tcat(lsa, lsb[iolb])
	RSR = tcat(rsa[iora], rsb)

	nlR = nlA + nolB
	nrR = norA + nrB

	nodA = length(odimsA)
	# indices into tcat(odimsA, odimsB)
	ldimsR = tcat(oneto(nlA), tupseq(nodA + 1, nodA + nolB))
	rdimsR = tcat(tupseq(nlA + 1, nlA + norA), tupseq(nodA + nolB + 1, nodA + nolB + nrB)) 
	#dimsR = tcat(itsa, iosa, nodA .+ oneto(nosB), (nodA + nosB) .+ itsb, tupseq(nA+1, nA+nosA), (nodA + nosB) .+ iosb)
	return :( ($tdimsA, $tdimsB, $odimsA, $odimsB, $ldimsR, $rdimsR, $LSR, $RSR) )
end


^(M::Tensor, x::Number) = Tensor(reshape(Matrix(M)^x, size(M)), Val(spaces(M)); checkspaces = false)
^(M::Tensor, x::Integer) = Tensor(reshape(Matrix(M)^x, size(M)), Val(spaces(M)); checkspaces = false)

#
# Analytic matrix functions
#

for f in [:inv, :sqrt, :exp, :log, :sin, :cos, :tan, :sinh, :cosh, :tanh]
	@eval function $f(M::Tensor)
			ensure_square(M)
			Tensor(reshape($f(Matrix(M)), size(M)), M)
		end
end


#
# Linear-algebra functions
#
det(M::Tensor) = begin ensure_square(M); det(Matrix(M)); end
opnorm(M::Tensor, args...) = begin ensure_square(M); opnorm(Matrix(M), args...); end

eigvals(M::Tensor, args...) = begin ensure_square(M); eigvals(Matrix(M), args...); end
svdvals(M::Tensor, args...) = svdvals(Matrix(M), args...)


# Broadcasting
struct MultiMatrixStyle{S,A,N} <: Broadcast.AbstractArrayStyle{N} end
MultiMatrixStyle{S,A,N}(::Val{N}) where {S,A,N} = MultiMatrixStyle{S,A,N}()

similar(bc::Broadcasted{MMS}, ::Type{T}) where {MMS<:MultiMatrixStyle{S,A,N}} where {S,N} where {A<:DenseArray} where {T} = similar(Tensor{S,T,N,A}, axes(bc))

BroadcastStyle(::Type{Tensor{S,T,N,A}}) where {S,T,N,A} = MultiMatrixStyle{S,A,N}()
BroadcastStyle(::Type{Tensor{S,T1,N,A1}}, ::Type{Tensor{S,T2,N,A2}})  where {A1<:DenseArray{T1,N}} where {A2<:DenseArray{T2,N}} where {S,N} where {T1,T2} = MultiMatrixStyle{S, promote_type(A1,A2), N}()
BroadcastStyle(::Type{<:Tensor}, ::Type{<:Tensor}) = error("To be broadcasted, MixedTensors must have the same dimensions and the same spaces in the same order")

# BroadcastStyle(::Type{BitString{L,N}}) where {L,N} = BitStringStyle{L,N}()
# BroadcastStyle(::BitStringStyle, t::Broadcast.DefaultArrayStyle{N}) where {N} = Broadcast.DefaultArrayStyle{max(1,N)}()


#
# """
# `*(A::Tensor, spaces...) * B::Tensor` contracts A with the specified subspaces of B,
# where `ndims(A) <= ndims(B)`, `length(spaces) = ndims(A)`, and `spaces ⊆ 1:ndims(B)`.
# This is equivalent to, but generally faster than, tensoring A with the identity operator, permuting, and multiplying with B.
#
# `A * (B, spaces)` contracts B along the specified subspaces of A.
# """
# function (*)(tup::Tuple{Tensor, Vararg{Int}}, B::Tensor)
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
# 	return Tensor(C_mat)
# end
#
# function mult(tup::Tuple{Tensor, Vararg{Int}}, B::Tensor)
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
# 	contract!(1, A.data, Val(:N), B.data, Val(:N), 0, new_data, oA, cA, oB, cB, iC, Val(:BLAS))
# 	return Tensor(new_data)
# end
#
#
# function (*)(A::Tensor, tup::Tuple{Tensor, TTuple{Int}})
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
# 	return Tensor(C_mat)
# end
#
#
# trace(X::Tensor) = trace(matrix(X))
# # Partial trace
# trace(X::Tensor, dims...) = trace(X, collect(dims))
# trace(X::Tensor, dims::Array{Int,1}) = trace_(X, dims, compl_dims(dims, ndims(X)))
#
# function trace_(X::Tensor, dims::Array{Int,1}, other_dims::Array{Int,1})
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
# 	return Tensor(new_data);
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
# reduce(X::Tensor, dims...) = reduce(X, collect(dims))
# reduce(X::Tensor, dims::Array{Int,1}) = trace_(X, compl_dims(collect(dims), ndims(X)), dims)




end
