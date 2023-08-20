"""
	AlgebraicTensors (module)

Implements tensors of arbitrary dimensionality and size as algebraic objects.
"""


#=
Function that take dimensions as inputs:
	size, axes, pernmutedims

Functions that take spaces as inputs:
	transpose
=#


#=
STATUS:
constructors 				WORKING, PERFORMANT
	> enable validation of ldims, rdims
	> implement means to bypass checking on the constructor

getindex						WORKING
	• still kind of slow due to unavoidable type instability.

setindex!					WORKINGm PERFORMANT

tr								WORKING, PERFORMANT

adjoint, transpose		WORKING, PERFORMANT

+,-							WORKING, PERFORMANT
	> generalize to multiple operands?

*,/							WORKING (Tensor * Tensor), IN PROGRESS (Tensor * Array)

^								WORKING, PERFORMANT

matrix ops					WORKING, PERFORMANT
(inv, exp, log, sin, cos, tan, sinh, cosh, tanh)

eigvals, svdvals			WORKING, PERFORMANT

eigvecs						WORKING, PERFORMANT
svd							WORKING, but results returned as NamedTuple instead of SVD
								(SVD requires U and Vt to have the same type, which they cannot)

broadcasting				NOT IMPLEMENTED


FEATURES TBD:
>	Would it make sense to define behavior of trace on non-square matrices>
>	Support in-place operations?
>	generalize + to >2 arguments
>	Use TensorIteration instead of @diagonal_op ?

Long term:
>	Implement (lazy) product tensors
>	Use Strided.jl?
>	Better support of different underlying array types?
			In places (e.g. *) it is assumed that the underlying array type has
			the element type as the first parameter and has a constructor of the
			form ArrayType{ElementType}(undef, size).
>	lperm/rperm or lspaces/rspaces instead of ldims/rdims?

=#

module AlgebraicTensors

export Tensor, lsize, rsize, spaces, lspaces, rspaces, nlspaces, nrspaces
export marginal
export tr, eigvals, svdvals, opnorm

using MiscUtils
using SuperTuples
using StaticArrays
using LinearAlgebra
using TensorOperations: trace!, contract!, tensoradd!, scalar
using Base.Broadcast: Broadcasted, BroadcastStyle
using Base: tail

import Base: display, show
import Base: ndims, length, size, axes, similar
import Base: reshape, permutedims, adjoint, transpose, Matrix, == #, isapprox
import Base: getindex, setindex!
import Base: (+), (-), (*), (/), (^)
import Base: inv, exp, log, sin, cos, tan, sinh, cosh, tanh

import LinearAlgebra: tr, eigvals, svdvals, opnorm, eigvecs, svd
import Base: similar
import SuperTuples.static_fn



#------------------------------------
# Preliminaries & utilities


const SpacesInt = UInt128  		# An integer treated as a bit set for vector spaces
const Iterable = Union{Tuple, AbstractArray, UnitRange, Base.Generator}
const Axes{N} = NTuple{N, AbstractUnitRange{<:Integer}}
const SupportedArray{T,N} = DenseArray{T,N}		# can change this later.  Maybe to StridedArray?

# Compute the dimensions of selected spaces after the others are contracted out
function remain_dims(dims::Dims{N}, ::Val{S}, ::Val{K}) where {N, S, K}
	# S and K should be SpacesInts
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
		#@warn "diagonal_op may not be correct for tensors with different left and right spaces"
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
struct Tensor{LS, RS, T, N, A<:SupportedArray{T,N}, NL, NR} <: SupportedArray{T,N}
	# LS and RS are wide unsigned integers whose bits indicate the left and right spaces
	# These are redundant with lspaces and rspaces, but are included as type parameters to
   # make tensor contraction performant.
	data::A
	lspaces::Dims{NL}
	rspaces::Dims{NR}

	# Primary inner constructor.
	function Tensor{LS, RS}(data::A, lspaces::NTuple{NL,Integer}, rspaces::NTuple{NR,Integer}; validate = true) where {LS, RS, NL, NR} where {A<:SupportedArray{T,N}} where {T,N}
		 LS isa SpacesInt && RS isa SpacesInt || error("Tensor parameters LS,RS must be of type SpacesInt")
		if validate
			NL + NR == N || error("ndims(A) == $N must be equal the total number of spaces (left + right) == $(NL+NR)")
			count_ones(LS) == NL || error("length(lspaces) must equal count_ones(LS)")
			count_ones(RS) == NR || error("length(rspaces) must equal count_ones(RS)")
			binteger(SpacesInt, lspaces) == LS || error("lspaces is inconsistent with parameter LS")
			binteger(SpacesInt, rspaces) == RS || error("rspaces is inconsistent with parameter RS")
		end
		return new{LS, RS, T, N, A, NL, NR}(data, lspaces, rspaces)
	end

	# Construct from Arrays whose type is not intrinsically supported
	Tensor{LS,RS}(data::AbstractArray, lspaces, rspaces; validate = true) where {LS,RS} = Tensor{LS,RS}(collect(data), lspaces, rspaces; validate)

	# # Shortcut constructor: construct from array, using another Tensor's metadata.
	# # By ensuring the array has the right number of dimensions, no input checking is needed.
	# function Tensor(arr::A_, M::Tensor{LS,RS,T,N,A,NL,NR}) where {A_ <: SupportedArray{T_,N}} where {T_} where {LS,RS,T,N,A,NL,NR}
	# 	return new{LS,RS,T_,N,A_,NL,NR}(arr, M.lspaces, M.rspaces)
	# end

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
Tensor(arr, lspaces, rspaces) = Tensor(arr, tuple(lspaces...), tuple(rspaces...))
Tensor(arr, lspaces::Dims, rspaces::Dims) = Tensor(arr, Val(lspaces), Val(rspaces))
function Tensor(arr, ::Val{lspaces}, ::Val{rspaces}) where {lspaces, rspaces} 
   LS = binteger(SpacesInt, Val(lspaces))
   RS = binteger(SpacesInt, Val(rspaces))  
   Tensor{LS,RS}(arr, lspaces, rspaces)
end

# Faster constructors when lspaces and rspaces are identical
Tensor(arr, spaces) = Tensor(arr, tuple(spaces...))
Tensor(arr, spaces::Dims) = Tensor(arr, Val(spaces))
function Tensor(arr, ::Val{lspaces}) where {lspaces} 
   LS = binteger(SpacesInt, Val(lspaces))
   Tensor{LS,LS}(arr, lspaces, lspaces)
end

Tensor(arr) = Tensor(arr, Val(oneto(ndims(arr))), Val(()))


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
(M::Tensor)(spaces::Dims) = Tensor(M.data, spaces)
(M::Tensor)(::Val{S}) where {S} = Tensor(M.data, Val(S), Val(S))

(M::Tensor)(lspaces, rspaces) where {N} = Tensor(M.data, tuple(lspaces...), tuple(rspaces...))  # for iterables
(M::Tensor)(lspaces::Dims, rspaces::Dims) where {N} = Tensor(M.data, lspaces, rspaces)


# Similar array with the same spaces and same size

# similar(M::Tensor{LS,RS,T,N,A}) where {LS,RS,T,N,A} = Tensor{LS,RS,T,N,A}(similar(M.data))
# # Similar array with different type, but same size and spaces.
# similar(M::Tensor{LS,RS,T,N}, newT) = Tensor{LS,RS,newT,N}(similar(M.data, newT), M.lperm, M.rperm)
#
# #Similar array with different type and/or size
# const Shape = Tuple{Union{Integer, Base.OneTo},Vararg{Union{Integer, Base.OneTo}}}
#
# similar(::Type{M}) where {M<:Tensor{LS,RS,T,N,A}} where {LS,RS} where {A<:SupportedArray{T,N}} where {T,N} = Tensor{LS,RS,T,N,A}(similar(A, shape))

#-------------
# Size and shape
ndims(M::Tensor) = ndims(M.data)


lspaces_int(M::Tensor{LS, RS}) where {LS,RS} = LS
rspaces_int(M::Tensor{LS, RS}) where {LS,RS} = RS

nlspaces(M::Tensor{LS, RS, T, N, A, NL, NR}) where {LS, RS, T, N, A, NL, NR} = NL
nrspaces(M::Tensor{LS, RS, T, N, A, NL, NR}) where {LS, RS, T, N, A, NL, NR} = NR

# Return the spaces in array order
spaces(M::Tensor) = (lspaces(M), rspaces(M))
lspaces(M::Tensor) = M.lspaces
rspaces(M::Tensor) = M.rspaces


length(M::Tensor) = length(M.data)

size(M::Tensor) = size(M.data)
size(M::Tensor, dim) = size(M.data, dim)
size(M::Tensor, dims::Iterable) = size(M.data, dims)

axes(M::Tensor) = axes(M.data)
axes(M::Tensor, dim) = axes(M.data, dim)
axes(M::Tensor, dims::Iterable) = map(d->axes(M.data, d), dims)

lsize(M::Tensor) = ntuple(i -> size(M.data, i), Val(nlspaces(M)))
rsize(M::Tensor) = ntuple(i -> size(M.data, i + nlspaces(M)), Val(nrspaces(M)))

lperm(M::Tensor) = sortperm(M.lspaces)
rperm(M::Tensor) = sortperm(M.rspaces)

# Not sure laxes and raxes are really necessary
laxes(M::Tensor) = ntuple(i -> axes(M.data,i), Val(nlspaces(M)))
raxes(M::Tensor) = ntuple(i -> axes(M.data, i + nlspaces(M)), Val(nrspaces(M)))


arraytype(::Tensor{LS,RS,T,N,A} where {LS,RS,T,N}) where A = A


# size2string(d) = isempty(d) ? "0-dim" :
#                  length(d) == 1 ? "length-$(d[1])" :
#                  join(map(string,d), '×')

display(M::Tensor) = show(M)

function show(io::IO, M::Tensor)
	# print as usual
	ls = join(map(string, lsize(M)), '×')
	rs = join(map(string, rsize(M)), '×')
	print(io, '(', ls, ")×(", rs, ") ")
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
	if X.lspaces == Y.lspaces && X.rspaces == Y.rspaces
		return X.data == Y.data
	end

	# check whether X,Y have the same elements when permuted
	# I tried combining the permutations so that only a single linear index was calculated,
	# but surprisingly that ended up being slower
	Xdims = (lperm(X)..., (rperm(X) .+ nlspaces(X))...)
	Ydims = (lperm(Y)..., (rperm(Y) .+ nlspaces(X))...)
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
	laxes(M)[lperm(M)] == raxes(M)[rperm(M)] ? M : throw(DimensionMismatch("Tensor is not square"))
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


# If accessing a single element, return that element.
"""
	getindex(T::Tensor, idx...)

Indexing a Tensor indexes the backing array.  If all the indices are scalars the result
is the specified array element.  Otherwise the result is a Tensor whose spaces are those
associated with the dimensions that were indexed by non-scalars.
"""
@inline getindex(M::Tensor, i::Vararg{Union{Integer, CartesianIndex}}) = getindex(M.data, i...)

# Index with ranges returns a Tensor.
# Spaces indexed by scalars are removed from the Tensor
@inline getindex(M::Tensor, idx...) = getindex_(M, idx)
@inline function getindex_(M::Tensor{LS,RS}, idx) where {LS,RS}	# for some reason, slurping/Vararg is slow
	# I = to_indices(M.data, idx)
	NL = nlspaces(M)
	NR = nrspaces(M)

	# Tuple of bools indicating which dimensions to keep
	ldmask = ntuple(i -> !isa(idx[i], Integer), Val(NL)) 
	rdmask = ntuple(i -> !isa(idx[i+NL], Integer), Val(NR))

   lspaces_ = lspaces(M)[ldmask]
   rspaces_ = rspaces(M)[rdmask]

	# LS_, RS_ are not inferrable because they depend on the run-time values lspaces, rspaces.
	# This slows down getindex a LOT 
   LS_ = binteger(SpacesInt, lspaces_)
   RS_ = binteger(SpacesInt, rspaces_)

	data_ = M.data[idx...]		# returning here is fast
	Tensor{LS_,RS_}(data_, lspaces_, rspaces_; validate = false)
end




@inline function setindex!(M::Tensor, data, idx...)
	M.data[idx...] = data
	M
end


@inline function setindex!(M::Tensor, S::Tensor, idx...)
	M.data[idx...] = S.data
	M
end

#-----------------------

reshape(M::Tensor, shape::Dims) = Tensor(reshape(M.data, shape), lspaces(M), rspaces(M))
permutedims(M::Tensor, ord) = Tensor(permutedims(M.data, ord), M)


# TODO: Make a lazy wrapper, just like Base does.
# Note, "partial adjoint" doesn't really make sense.
function adjoint(M::Tensor{LS,RS}) where {LS,RS}
	NL = nlspaces(M)
	NR = nrspaces(M)
   perm = (tupseq(NL+1, NL+NR)..., oneto(Val{NL})...)
	# adjoint is called element-by-element (i.e. it recurses as required)
	return Tensor{RS,LS}(permutedims(adjoint.(M.data), perm), M.rspaces, M.lspaces; validate = false)
end


#Swap left and right dimensions of a Tensor
"""
Tensor transpose and partial transpose.

	transpose(T::Tensor)

transposes (exchanges the left and right spaces) of `T`. 

	transpose(T::Tensor, space)
	transpose(T::Tensor, spaces...)

Transpose selected spaces. If `T` does not involve the specified spaces
the operation simply returns `T`.

Note: the order of the spaces on output is not defined and should not be relied upon.
"""
function transpose(M::Tensor{LS,RS}) where {LS,RS}
	NL = nlspaces(M)
	NR = nrspaces(M)
   perm = (tupseq(NL+1, NL+NR)..., oneto(Val{NL})...)
	return Tensor{RS,LS}(permutedims(M.data, perm), M.rspaces, M.lspaces; validate = false)
end



# function unindexed_dims(dims::Dims{N}, dkeep, ::Val{S}) where {N, S}
# 	(S isa SpacesInt) || error("S must be a SpacseInt")
# 	count_ones(S) == N || error("count_ones(S) must equal length(dims)")

# 	spaces = findnzbits(Val(S))
# 	skeep = findin(dims, dkeep)
# 	spaces_ = spaces[skeep]
# 	dims_ = sortperm(spaces_)
# 	S_ = binteger(SpacesInt, spaces_)
# 	return (S_, dims_)
# end


##  RESUME WORKING HERE!!!

# Partial transpose
"""
	transpose(tensor, space)
	transpose(tensor, spaces)
Toggle the "direction" (left or right) of specified tensor spaces. Any `spaces` that are not
present in `tensor` are ignored.  Untransposed spaces and transposed spaces involving both 
a left and right version retain their relative order.  Transposed spaces for which there is
only a left or only a right side are placed after the other spaces.
"""
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
		tlperm = lperm(M)[it_lspaces]			# left dimensions to be transposed (in space order)
		trperm = rperm(M)[it_rspaces]		# right dimensions to be transposed (in space order)
		ldims_ = setindex(lperm(M), trperm, it_lspaces)		# the transposed spaces have the same order
		rdims_ = setindex(rperm(M), tlperm, it_rspaces) .+ nlspaces(A)
		arr = permutedims(M.data, (ldims_..., rdims_...))
		return Tensor{LS,RS}(arr, M.lspaces, M.rspaces)
	else

		# new left and right spaces
		LS_ = (LS & ~TS) | TSR
		RS_ = (RS & ~TS) | TSL
		NL_ = count_ones(LS_)
		NR_ = count_ones(RS_)

		TSB = TSL & TSR		# transposed spaces common to both sides
		TSL = TSL ⊻ TSB		# transposed spaces on left only
		TSR = TSR ⊻ TSB		# transposed spaces on right only

		lperm = sortperm(lspaces(M))
		rperm = sortperm(rspaces(M))

		ldims = invperm(lspaces(M), lperm)
		rdims = invperm(rspaces(M), rperm)

		it_ldims = findnzbits(TSB, LS)		# indices of left spaces to be transposed
		it_rdims = findnzbits(TSB, RS)		# indices of right spaces to be transposed

		ldims_ = setindex(ldims, rdims[it_rdims], it_ldims)
		rdims_ = setindex(rdims, ldims[it_ldims], it_rdims)

		lperm = ldims_[lperm]
		rperm = rdims_[rperm]

		error("RESUME WORK HERE")
		lkeep = findnzbits(LS ⊻ TSL, LS)[lperm]
		rkeep = findnzbits(RS ⊻ TSR, RS)[rperm]


		# ik_lspaces = findnzbits(~TSB, LS)		# indices of left spaces to keep as left
		# ik_rspaces = findnzbits(~TSB, RS)		# indices of right spaces to keep as right


		# lsp = findnzbits(LS)
		# rsp = findnzbits(RS)

		# lspaces_ = (lsp[ik_lspaces]..., rsp[it_rspaces]...)
		# lperm_ = sortperm(lspaces_)
		# ldims_ = (M.ldims[ik_lspaces]..., M.rdims[it_rspaces]...)		# dims of the new left spaces
		# lperm = ldims_[lperm_];

		# rspaces_ = (rsp[ik_rspaces]..., lsp[it_lspaces]...)
		# rperm_ = sortperm(rspaces_)
		# rperm = (M.rdims[ik_rspaces]..., M.ldims[it_lspaces]...)[rperm_];

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
   ensure_square(M)
   s = zero(eltype(M))
	# this is much faster than using TensorOperations.trace!
   @diagonal_op M iA s += M.data[iA]
   return s
end

# function tr2(M::Tensor{LS,LS}) where {LS}
# 	T = eltype(M)
# 	scalar(trace!(one(T), M.data, :N, zero(T), Array{T,0}(undef), (), (), M.ldims, M.rdims))
# end

# Partial trace.  It's not convenient to use a keyword because it clobbers the method
# that doesn't have a spaces argument.
tr(M::Tensor, space::Integer) = tr(M, (space,))
tr(M::Tensor, spaces::Dims) = tr(M, Val(spaces))

function tr(M::Tensor{LS,RS}, ::Val{tspaces}) where {LS,RS,tspaces}
	# Int representation of traced spaces
	if tspaces isa Dims
		TS = binteger(SpacesInt, Val(tspaces))
	elseif tspaces isa SpacesInt
		TS = tspaces
	else
		error("tspaces must be a SpacesInt or Dims")
	end

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


	ltdims = M.ldims[findnzbits(TS, LS)]
	rtdims = M.rdims[findnzbits(TS, RS)]

	lkdims = M.ldims[findnzbits(KLS, LS)]
	rkdims = M.rdims[findnzbits(KRS, RS)]

	NL = length(lkdims)
	NR = length(rkdims)
	N = NL + NR
	sz = size(M.data)
	R = similar(M.data, (sz[lkdims]..., sz[rkdims]...))
	T = eltype(M)
	trace!(one(T), M.data, :N, zero(T), R, lkdims, rkdims, ltdims, rtdims)
	return Tensor{KLS, KRS}(R, oneto(NL), tupseq(NL+1, NL+NR))
end

"""
	marginal(T::Tensor, spaces)

Trace out all but the specified spaces.
"""
marginal(M::Tensor, space::Integer) = marginal(M, (space,))
marginal(M::Tensor, spaces::Dims) = marginal(M, Val(spaces))

function marginal(M::Tensor{LS,RS}, ::Val{kspaces}) where {LS,RS,kspaces}
	KS = binteger(SpacesInt, Val(kspaces))
	TSL = ~KS & LS
	TSR = ~KS & RS
	TSL == TSR || error("trace would act upon unequal left and right spaces")
	return tr(M, Val(TSL))
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


# Addition and subtraction

-(M::Tensor) = Tensor(-M.data, M)

# fallback (for unequal spaces)
+(A::Tensor, B::Tensor) = error("Can only add or subtract Tensors with the same spaces")
-(A::Tensor, B::Tensor) = error("Can only add or subtract Tensors with the same spaces")


function +(A::Tensor{LS,RS}, B::Tensor{LS,RS}) where {LS,RS}
	if A.ldims == B.ldims && A.rdims == B.rdims
		# Same spaces in the same order
		return Tensor(A.data + B.data, A)
	else
		# Same spaces in different order
		# Same code as in ==. Should we make a macro?
		Adims = (A.ldims..., A.rdims...)
		Bdims = (B.ldims..., B.rdims...)

		BinA = Bdims[invperm(Adims)]

		Rdata = A.data + permutedims(B.data, BinA)
		return Tensor(Rdata, A)
	end
end


function -(A::Tensor{LS,RS}, B::Tensor{LS,RS}) where {LS,RS}
	if A.ldims == B.ldims && A.rdims == B.rdims
		# Same spaces in the same order
		return Tensor(A.data - B.data, A)
	else
		# Same spaces in different order
		# Same code as in ==. Should we make a macro?
		Adims = (A.ldims..., A.rdims...)
		Bdims = (B.ldims..., B.rdims...)

		BinA = Bdims[invperm(Adims)]

		Rdata = A.data - permutedims(B.data, BinA)
		return Tensor(Rdata, A)
	end
end

#    # This is slightly slower
# 	function +(A::Tensor{LS,RS}, B::Tensor{LS,RS}) where {LS,RS}
#    NL = nlspaces(A)
#    NR = nrspaces(A)

#    # Same code as in ==. Should we make a macro?
#    Adims = (A.ldims..., A.rdims...)
#    Bdims = (B.ldims..., B.rdims...)
#    Astrides = calc_strides(axes(A))[Adims]
#    Bstrides = calc_strides(axes(B))[Bdims]

#    axR = axes(A)
#    Rtype = promote_type(arraytype(A), arraytype(B))
#    szR = map(a->last(a)-first(a)+1, axR)
#    Rdata = Rtype(undef, szR)
   
#    @inbounds for ci in CartesianIndices(axR)
#    	iA = calc_index(ci, Astrides)
#    	iB = calc_index(ci, Bstrides)
#    	Rdata[ci] = A[iA] + B[iB]
#    end

#    return Tensor{LS,RS}(Rdata, oneto(NL), tupseq(NL+1, NL+NR))
# end



#  Multiplication

*(M::Tensor, x::Number) = Tensor(M.data * x, M)
*(x::Number, M::Tensor) = Tensor(x * M.data, M)
/(M::Tensor, x::Number) = Tensor(M.data / x, M)

# Fallbacks  -- should we have these?
# *(M::Tensor, x) = Tensor(M.data * x, M)
# *(x, M::Tensor) = Tensor(x * M.data, M)
# /(M::Tensor, x) = Tensor(M.data / x, M)


# Matrix multiplication.  These methods are actually quite fast -- about as fast as the
# core matrix multiplication. We appear to incur very little overhead.



# !!!
# Is this really desirable behavior?  Perhaps it would make more sense to have
# the right dims of M contract again all the dims of A.

"""
`T*X` where `T` is a Tensor and `X` is an `AbstractArray` contracts the right
dimensions of `T` with dimensions `rspaces(T)` of `X`.  The result is an array of type
similar to `X`, whose size along the contracted dimensions is `lsize(T)` and whose size in
the uncontracted dimensions is that of `X`.

`X*T` is similar, except the left dimensions of `T` are contracted against `X`, and the
size of the result depends on the `rsize(M)`.
"""
*(M::Tensor, A::SupportedArray{TA,1}) where {TA} = _mult_MA(M, A)
*(M::Tensor, A::SupportedArray{TA,2}) where {TA} = _mult_MA(M, A)
*(M::Tensor, A::SupportedArray{TA}) where {TA} = _mult_MA(M, A)

function _mult_MA(M::Tensor, A::SupportedArray)
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
#*(M::Tensor, A::SupportedArray{TA,1}) where {TA} =


*(A::SupportedArray{TA,1}, M::Tensor) where {TA} = _mult_AM(A, M)
*(A::SupportedArray{TA,2}, M::Tensor) where {TA} = _mult_AM(A, M)
*(A::SupportedArray{TA}, M::Tensor) where {TA} = _mult_AM(A, M)

# This is almost identical to the M*A version.
function _mult_AM(A::SupportedArray, M::Tensor)
	error("The desired behavior of Array * Tensor has not been decided.")

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
Tensor-Tensor multiplication.

	C = A*B

contracts the right spaces of Tensor `A` with matching left spaces of Tensor `B`.
The left spaces of `C` are the left spaces of `A` and the uncontracted left spaces of `B`,
provided these are distinct. (An error is thrown if they are not.)  Similarly, the right
spaces of `C` are the right spaces of `B` and the uncontracted right spaces of `B`, provided
these are distinct.  The order of the output spaces is undefined and should not be relied upon.
"""
function *(A::Tensor{LSA,CS}, B::Tensor{CS,RSB}) where {LSA,RSB,CS}
	# All the right spaces of A contract with all the left spaces of B.
	# All we need to do is put them in the correct order
	axes(A, A.rdims) == axes(B, B.ldims) || throw(DimensionMismatch("The right axes of A must be the same as the correponding left axes of B"))

	szR = (lsize(A)..., rsize(B)...)

	NLA = nlspaces(A)
	NRA = nrspaces(A)
	NLB = nlspaces(B)
	NRB = nrspaces(B)
	ldims_ = A.ldims
	rdims_ = B.rdims .+ (NLB - NLA)

	if rspaces(A) == lspaces(B)
		# the spaces are the same and in the same order.
		# Multiply as matrices (faster than tensor contraction)
		R = reshape(Matrix(A) * Matrix(B), (lsize(A)..., rsize(B)...))
	else
		# the spaces are in different orders.  Use tensor contraction
		TR = promote_type(eltype(A), eltype(B))
		R = Array{TR}(undef, szR)
		contract!(one(TR), A.data, :N, B.data, :N, zero(TR), R, 
			oneto(NLA), A.rdims, tupseq(NLB+1, NLB+NRB), B.ldims, oneto(NLA), tupseq(NLA+1, NLA+NRB), nothing)
	end
	Tensor{LSA,RSB}(R, oneto(NLA), tupseq(NLA+1, NLA+NRB))
end





function *(A::Tensor{LSA,RSA}, B::Tensor{LSB,RSB}) where {LSA,RSA} where {LSB,RSB}
	# Check for compatibility of spaces
	CS = RSA & LSB			# spaces to be contracted
	URSA = RSA & (~CS)		# right spaces of A, not contracted
	ULSB = LSB & (~CS)		# left spaces of B, not contracted

	LSA & ULSB == SpacesInt(0) || error("A and B have uncontracted left spaces in common")
	RSB & URSA == SpacesInt(0) || error("A and B have uncontracted right spaces in common")

	# left and right spaces of the result
	LSR = LSA | ULSB
	RSR = RSB | URSA

	# determine which dimensions are contracted and which are not, and their order
	# (needed for TensorOperations.contract!)
	tdimsA = A.rdims[findnzbits(CS, RSA)]
	tdimsB = B.ldims[findnzbits(CS, LSB)]
	axes(A, tdimsA) == axes(B, tdimsB) || throw(DimensionMismatch("raxes(A) must equal laxes(B) on spaces common to A,B"))

	urdimsA = A.rdims[findnzbits(URSA, RSA)]		# uncontracted right dims of A
	uldimsB = B.ldims[findnzbits(ULSB, LSB)]		# uncontracted left dims of B
	odimsA = (A.ldims..., urdimsA...)				# all "open" (uncontracted) dims of A
	odimsB = (uldimsB..., B.rdims...)					# all "open" (uncontracted) dims of B
	
	# order of dimensions of R:
	#   (left dims of A, uncontracted left dims of B, uncrontracted right dims of A, right dims of B)
	
	NLA = length(A.ldims)
	NOA = NLA + length(urdimsA)
	NOB = length(uldimsB) + length(B.rdims)

	lorder = sortperm((findnzbits(LSA)..., findnzbits(ULSB)...))
	rorder = sortperm((findnzbits(URSA)..., findnzbits(RSB)...))

	# The dims of R will be in the order of the spaces
	ldimsRo = (oneto(NLA)..., tupseq(NOA+1, NOA + length(uldimsB))...)[lorder]
	rdimsRo = (tupseq(NLA + 1, NLA + length(urdimsA))..., tupseq(NOA + length(uldimsB) + 1, NOA + NOB)...)[rorder]

	# ldimsR = sortperm(lspacesR)
	# rdimsR = sortperm(rspacssR)
	# println("LSR, RSR = ", findnzbits(LSR), findnzbits(RSR))
	# println("tdimsA = ", tdimsA)
	# println("odimsA = ", odimsA)
	# println("tdimsB = ", tdimsB)
	# println("odimsB = ", odimsB)
	# println("ldimsR = ", ldimsR)
	# println("rdimsR = ", rdimsR)
	# println("rdimsR = ", rdimsR, " = ", tupseq(NLA + 1, NLA + length(urdimsA)), tupseq(NOA + length(uldimsB) + 1, NOA + NOB))

	# allocate the output array
	szR = (size(A, A.ldims)..., size(B, uldimsB)..., size(A, urdimsA)..., size(B, B.rdims)...)
	TR = promote_type(eltype(A), eltype(B))
	R = Array{TR}(undef, szR)

	# contract
	contract!(one(TR), A.data, :N, B.data, :N, zero(TR), R,
						odimsA, tdimsA, odimsB, tdimsB, ldimsRo, rdimsRo, nothing)

	NLR = count_ones(LSR)		# should equal	NLA + length(uldimsB)
	NRR = count_ones(RSR)		# should equal length(urdimsA) + length(B.rdims)
	return Tensor{LSR,RSR}(R, oneto(NLR), tupseq(NLR+1, NLR+NRR))
	#return nothing
end



# Exponentiation

^(M::Tensor, x::Number) = Tensor(reshape(Matrix(M)^x, size(M)), M)
^(M::Tensor, x::Integer) = Tensor(reshape(Matrix(M)^x, size(M)), M)


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

function eigvecs(M::Tensor{S}, args...) where {S}
	ensure_square(M)
	sz = (lsize(M)..., prod(rsize(M)))
	V = reshape(eigvecs(Matrix(M), args...), sz)
	return Tensor{S, SpacesInt(1)}(V, M.ldims, (nlspaces(M)+1,))
end


function svd(M::Tensor{LS}, args...) where {LS}
	ensure_square(M)
	msize = lsize(M);
	nsvd = prod(rsize(M))
	ldims = M.ldims;
	rdims = (nlspaces(M)+1,)
	RS = SpacesInt(1) 
	result = svd(Matrix(M), args...)
	U = Tensor{LS, RS}(reshape(result.U, (msize..., nsvd)), ldims, rdims)
	Vt = Tensor{RS, LS}(reshape(result.Vt, (nsvd, msize...)), (1,), ldims .+ 1)
	return (U = U, S = result.S, Vt = Vt)
end

#
# Broadcasting
#

struct MultiMatrixStyle{S,A,N} <: Broadcast.AbstractArrayStyle{N} end
MultiMatrixStyle{S,A,N}(::Val{N}) where {S,A,N} = MultiMatrixStyle{S,A,N}()

similar(bc::Broadcasted{MMS}, ::Type{T}) where {MMS<:MultiMatrixStyle{S,A,N}} where {S,N} where {A<:SupportedArray} where {T} = similar(Tensor{S,T,N,A}, axes(bc))

BroadcastStyle(::Type{Tensor{S,T,N,A}}) where {S,T,N,A} = MultiMatrixStyle{S,A,N}()
BroadcastStyle(::Type{Tensor{S,T1,N,A1}}, ::Type{Tensor{S,T2,N,A2}})  where {A1<:SupportedArray{T1,N}} where {A2<:SupportedArray{T2,N}} where {S,N} where {T1,T2} = MultiMatrixStyle{S, promote_type(A1,A2), N}()
BroadcastStyle(::Type{<:Tensor}, ::Type{<:Tensor}) = error("To be broadcasted, Tensors must have the same dimensions and the same spaces in the same order")

# BroadcastStyle(::Type{BitString{L,N}}) where {L,N} = BitStringStyle{L,N}()
# BroadcastStyle(::BitStringStyle, t::Broadcast.DefaultArrayStyle{N}) where {N} = Broadcast.DefaultArrayStyle{max(1,N)}()




end
