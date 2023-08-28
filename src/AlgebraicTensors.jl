"""
	AlgebraicTensors (module)

Implements tensors of arbitrary dimensionality and size as algebraic objects.
"""


#=
8/20/2023
	This is a major revision to (1) make getindex type stable and (2) use the new TensorOperation.jl API.
	I am going to return to the original idea parameterizing the type with tuples for the ordered left and
	right spaces. Hopefully compiler improvements have made this inferrable now.
	This should make all types functions fast, but may result in excessive code generation.
=#

#=
Constructors	Done, performant
getindex()		Done, performant
setindex!()		Done
==()				Done
Tensor*Tensor	Done, performant


TODO:
- fix findlspace, findrspaces order of odims
- trace
- transpose, adjoint
- +,-
- Tensor*Array?
- broadcasting
- lazy tensor products
- better mechanism for comparison, element-wise ops?
- test performance in non-inferrable cases
=#


module AlgebraicTensors

export Tensor, lsize, rsize, spaces, lspaces, rspaces, nlspaces, nrspaces
export marginal
export tr, eigvals, svdvals, opnorm
export tr_jutho, tr_mine

using MiscUtils
using SuperTuples
using StaticArrays
using LinearAlgebra
using TensorOperations: Index2Tuple, tensortrace, tensorcontract, tensoradd, tensorscalar
using Base.Broadcast: Broadcasted, BroadcastStyle
import Base: display, show
import Base: ndims, length, size, axes, similar, tail
import Base: reshape, permutedims, adjoint, transpose, Matrix, == #, isapprox
import Base: similar
import Base: getindex, setindex!
using Base: tail
import Base: (+), (-), (*), (/), (^)
import Base: inv, exp, log, sin, cos, tan, sinh, cosh, tanh
import LinearAlgebra: tr, eigvals, svdvals, opnorm, eigvecs, svd



#------------------------------------
# Definitions

const SpacesInt = UInt128  		# An integer treated as a bit set for vector spaces
const Spaces{N} = Tuple{Vararg{Integer,N}}
const Iterable = Union{Tuple, AbstractArray, UnitRange, Base.Generator}
const Axes{N} = NTuple{N, AbstractUnitRange{<:Integer}}
const SupportedArray{T,N} = DenseArray{T,N}		# can change this later.  Maybe to StridedArray?



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
		if lspaces($escM) == rspaces($escM)
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
struct Tensor{LS, RS, T, N, A<:SupportedArray{T,N}} <: SupportedArray{T,N}
	# LS and RS are tuples of the left and right spaces associated with the dimensions of the array
	data::A

	# Primary inner constructor.
	"""
		Tensor{lspaces,rspaces}(A::AbstractArray, lspaces::Tuple, rspaces::Tuple)

	Create from array 'A' a Tensor acting on left and right spaces given by tuples
	'lspaces' and 'rspaces'. 
	The first `length(lspaces)` dimensions of `A` correspond the left spaces.
	The last `length(rspaces)` dimensions of `A` correspond to the right spaces.

	Tensor{spaces}(A::AbstractArray)

	creaes a tensor with identical left and right spaces.
	
		Tensor(A::AbstractArray)
	
	creates a left vector with `lspaces = 1:ndims(A)` and `rspaces = ()`.
	"""
	function Tensor{LS_, RS_}(data::A; validate = true) where {LS_, RS_} where {A<:SupportedArray{T,N}} where {T,N}
		if validate
			LS = isa(LS_, Dims) ? LS_ : tuple(LS_...)
			RS = isa(RS_, Dims) ? RS_ : tuple(RS_...)

			nbits = sizeof(SpacesInt)*8
			all(LS .> 0) && all(LS .<= nbits) || error("Values of LSPACES must be integer between 1 and $nbits")
			all(RS .> 0) && all(RS .<= nbits) || error("Values of RSPACES must be integer between 1 and $nbits")
			LSint = binteger(SpacesInt, Val(LS))
			RSint = binteger(SpacesInt, Val(RS))
			NL = length(LS)
			NR = length(RS)
			count_ones(LSint) == NL || error("LSPACES has repeated values")
			count_ones(RSint) == NR || error("RSPACES has repeated values")
			NL + NR == N || error("ndims(A) == $N must equal the total number of spaces (left + right) == $(NL+NR)")
		else
			LS = LS_
			RS = RS_
		end
		return new{LS, RS, T, N, A}(data)
	end

	# Construct from Arrays whose type is not intrinsically supported
	Tensor{LS,RS}(data::AbstractArray; validate = true) where {LS,RS} = Tensor{LS,RS}(collect(data); validate)

	# # Shortcut constructor: construct from array, using another Tensor's metadata.
	# # By ensuring the array has the right number of dimensions, no input checking is needed.
	# function Tensor(arr::A_, M::Tensor{LS,RS,T,N,A,NL,NR}) where {A_ <: SupportedArray{T_,N}} where {T_} where {LS,RS,T,N,A,NL,NR}
	# 	return new{LS,RS,T_,N,A_,NL,NR}(arr, M.lspaces, M.rspaces)
	# end

end



# Convenience constructors

Tensor{LS}(arr) where {LS} = Tensor{LS, LS}(arr)
Tensor(arr) = Tensor{oneto(ndims(arr)), ()}(arr)


# Reconstruct Tensor with different spaces
"""
	(T::Tensor)(spaces...)
	(T::Tensor)(spaces::Tuple)
	(T::Tensor)(lspaces::Tuple, rspaces::Tuple)`

Create a Tensor with the same data as `M` but acting on different spaces.
(This is a lazy operation that is generally to be preferred over [`permutedims`](@ref).)
"""
(M::Tensor)(spaces::Vararg{Int64}) = M(spaces,spaces)						# for list of ints
(M::Tensor)(spaces) = Tensor{tuple(spaces...)}(M.data)			# for iterable
(M::Tensor)(spaces::Dims) = Tensor{spaces,spaces}(M.data)
(M::Tensor)(::Val{S}) where {S<:Dims} = Tensor{S}(M.data)
(M::Tensor)(lspaces, rspaces) = Tensor{tuple(lspaces...), tuple(rspaces...)}(M.data)  # for iterables
(M::Tensor)(lspaces::Dims, rspaces::Dims) = Tensor{lspaces, rspaces}(M.data)
(M::Tensor)(::Val{LS}, ::Val{RS}) where {LS, RS} = Tensor{LS, RS}(M.data)


# Similar array with the same spaces and same size

# similar(M::Tensor{LS,RS,T,N,A}) where {LS,RS,T,N,A} = Tensor{LS,RS,T,N,A}(similar(M.data))
# # Similar array with different type, but same size and spaces.
# similar(M::Tensor{LS,RS,T,N}, newT) = Tensor{LS,RS,newT,N}(similar(M.data, newT), M.lperm, M.rperm)
#
# #Similar array with different type and/or size
# const Shape = Tuple{Union{Integer, Base.OneTo},Vararg{Union{Integer, Base.OneTo}}}
#
# similar(::Type{M}) where {M<:Tensor{LS,RS,T,N,A}} where {LS,RS} where {A<:SupportedArray{T,N}} where {T,N} = Tensor{LS,RS,T,N,A}(similar(A, shape))


#-------------------------
# Size and shape

ndims(M::Tensor) = ndims(M.data)

lspaces(M::Tensor{LS,RS}) where {LS, RS} = LS
rspaces(M::Tensor{LS,RS}) where {LS,RS} = RS
spaces(M::Tensor) = (lspaces(M)..., rspaces(M)...)

nlspaces(M) = length(lspaces(M))
nrspaces(M) = length(rspaces(M))

length(M::Tensor) = length(M.data)

size(M::Tensor, args...) = size(M.data, args...)
lsize(M::Tensor) = ntuple(i -> size(M.data, i), nlspaces(M))
rsize(M::Tensor) = ntuple(i -> size(M.data, i + nlspaces(M)), nrspaces(M))

axes(M::Tensor, args...) = axes(M.data, args...)
laxes(M::Tensor) = ntuple(i -> axes(M.data,i), nlspaces(M))
raxes(M::Tensor) = ntuple(i -> axes(M.data, i + nlspaces(M)), nrspaces(M))


# utitlity functions
lspaces_int(M::Tensor) = binteger(SpacesInt, Val(lspaces(M)))
rspaces_int(M::Tensor) = binteger(SpacesInt, Val(rspaces(M)))

lperm(M::Tensor) = sortperm(lspaces(M))
rperm(M::Tensor) = sortperm(rspaces(M))


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


#-----------------------------------------
# Comparison

# X and Y have the same spaces
function ==(X::Tensor, Y::Tensor)

	if nlspaces(X) != nlspaces(Y) || nrspaces(X) != nrspaces(Y)
		return false
	end

		# fast comparison if X,Y have same spaces in the same order
	if lspaces(X) == lspaces(Y) && rspaces(X) == rspaces(Y)
		return X.data == Y.data
	end

	XLS_ = lspaces(X)(lperm(X))
	XRS_ = rspaces(X)(rperm(X))
	YLS_ = lspaces(Y)(lperm(Y))
	YRS_ = rspaces(Y)(rperm(Y))
	
	 # Comparison if X,Y have the same spaces in different order
	if XLS_ == YLS_ && XRS_ == YRS_
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

	# Different spaces
	return false
end


# Internal functions to make ensure a Tensor is "square".
# A Tensor M is *square* if it has the same left and right spaces (in any order)
# and the corresponding left and right axes are the same.
# It is "proper" square if it is square and the left and right spaces are in the same order.
#
# ensure_square(M) throws an error if M is not square; otherwise it returns nothing
@inline function ensure_square(M::Tensor)
	lp = lperm(M)
	rp = rperm(M)
	is_square = (lspaces(M)[lp] == rspaces(M)[rp]) && (laxes(M)[lp] == raxes(M)[rp])
	is_square ? nothing : throw(DimensionMismatch("Tensor is not square"))
end




# Conversions
convert(T, M::Tensor) = convert(T, M.data)


"""
	Matrix(M::Tensor)

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

   LS_ = lspaces(M)[ldmask]
   RS_ = rspaces(M)[rdmask]

	data_ = M.data[idx...]		# returning here is fast
	Tensor{LS_,RS_}(data_; validate = false)
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

# Reshape does not support changing the number of dimensions of the underlying array,
# since the associated spaces would be ambiguous.
reshape(M::Tensor{LS,RS}, shape::Dims) where {LS,RS} = Tensor{LS,RS}(reshape(M.data, shape))
permutedims(M::Tensor, ord) = Tensor(permutedims(M.data, ord), M)


# TODO: Make a lazy wrapper, just like Base does.
# Note, "partial adjoint" doesn't really make sense.
function adjoint(M::Tensor{LS,RS}) where {LS,RS}
	NL = nlspaces(M)
	NR = nrspaces(M)
   perm = (tuplerange(NL+1, NL+NR)..., oneto(NL)...)
	# adjoint is called element-by-element (i.e. it recurses as required)
	return Tensor{RS,LS}(permutedims(adjoint.(M.data), perm); validate = false)
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
   perm = (tuplerange(NL+1, NL+NR)..., oneto(NL)...)
	return Tensor{RS,LS}(permutedims(M.data, perm); validate = false)
end




# # Partial transpose
# """
# 	transpose(tensor, space)
# 	transpose(tensor, spaces)
# Toggle the "direction" (left or right) of specified tensor spaces. Any `spaces` that are not
# present in `tensor` are ignored.  Untransposed spaces and transposed spaces involving both 
# a left and right version retain their relative order.  Transposed spaces for which there is
# only a left or only a right side are placed after the other spaces.
# """
# transpose(M::Tensor, ts::Int) = transpose(M, Val((ts,)))
# transpose(M::Tensor, ts::Dims) = transpose(M, Val(ts))
# function transpose(M::Tensor{LS,RS}, ::Val{tspaces}) where {LS, RS, tspaces}
# 	NL = nlspaces(M)
# 	NR = nrspaces(M)

# 	TS = binteger(SpacesInt, Val(tspaces))
# 	TSL = TS & LS		# transposed spaces on the left side
# 	TSR = TS & RS		# transposed spaces on the right side


# 	if iszero(TSL) && iszero(TSR)
# 		# nothing to transpose
# 		return M
# 	elseif TSL == TS && TSR == TS
# 		# All the spaces are in both left and right.
# 		# We just need to permute the dimensions, the spaces stay the same

# 		it_lspaces = findnzbits(TS, LS)		# indices of left spaces to be transposed (in space order)
# 		it_rspaces = findnzbits(TS, RS)		# indices of right spaces to be transposed (in space order)
# 		tlperm = lperm(M)[it_lspaces]			# left dimensions to be transposed (in space order)
# 		trperm = rperm(M)[it_rspaces]		# right dimensions to be transposed (in space order)
# 		ldims_ = setindex(lperm(M), trperm, it_lspaces)		# the transposed spaces have the same order
# 		rdims_ = setindex(rperm(M), tlperm, it_rspaces) .+ nlspaces(A)
# 		arr = permutedims(M.data, (ldims_..., rdims_...))
# 		return Tensor{LS,RS}(arr, M.lspaces, M.rspaces)
# 	else

# 		# new left and right spaces
# 		LS_ = (LS & ~TS) | TSR
# 		RS_ = (RS & ~TS) | TSL
# 		NL_ = count_ones(LS_)
# 		NR_ = count_ones(RS_)

# 		TSB = TSL & TSR		# transposed spaces common to both sides
# 		TSL = TSL ⊻ TSB		# transposed spaces on left only
# 		TSR = TSR ⊻ TSB		# transposed spaces on right only

# 		lperm = sortperm(lspaces(M))
# 		rperm = sortperm(rspaces(M))

# 		ldims = invperm(lspaces(M), lperm)
# 		rdims = invperm(rspaces(M), rperm)

# 		it_ldims = findnzbits(TSB, LS)		# indices of left spaces to be transposed
# 		it_rdims = findnzbits(TSB, RS)		# indices of right spaces to be transposed

# 		ldims_ = setindex(ldims, rdims[it_rdims], it_ldims)
# 		rdims_ = setindex(rdims, ldims[it_ldims], it_rdims)

# 		lperm = ldims_[lperm]
# 		rperm = rdims_[rperm]

# 		error("RESUME WORK HERE")
# 		lkeep = findnzbits(LS ⊻ TSL, LS)[lperm]
# 		rkeep = findnzbits(RS ⊻ TSR, RS)[rperm]


# 		# ik_lspaces = findnzbits(~TSB, LS)		# indices of left spaces to keep as left
# 		# ik_rspaces = findnzbits(~TSB, RS)		# indices of right spaces to keep as right


# 		# lsp = findnzbits(LS)
# 		# rsp = findnzbits(RS)

# 		# lspaces_ = (lsp[ik_lspaces]..., rsp[it_rspaces]...)
# 		# lperm_ = sortperm(lspaces_)
# 		# ldims_ = (M.ldims[ik_lspaces]..., M.rdims[it_rspaces]...)		# dims of the new left spaces
# 		# lperm = ldims_[lperm_];

# 		# rspaces_ = (rsp[ik_rspaces]..., lsp[it_lspaces]...)
# 		# rperm_ = sortperm(rspaces_)
# 		# rperm = (M.rdims[ik_rspaces]..., M.ldims[it_lspaces]...)[rperm_];

# 		perm = (lperm..., rperm...)
		
# 		arr = permutedims(M.data, perm)
# 		return Tensor{LS_,RS_}(arr, oneto(NL_), tupseq(NL_+1, NL_+NR_))

# 	end
# end


# TODO: Resume here

"""
	tr(A)

The trace of tensor `A`, which must be square.  Each left space of `A`
is contracted with the corresponding right space, yieldig a scalar.

	tr(A, spaces)

Trace over the the indicated spaces, returning another `Tensor` (even if the result is a scalar).
`spaces` can be an integer or tuple.
"""
function tr(M::Tensor)
	LS = lspaces_int(M)
	RS = rspaces_int(M)
	LS == RS || error("To trace a tensor, it must have matching left and right spaces")
	
	if lspaces(M) == rspaces(M)
		ensure_square(M)
   		s = zero(eltype(M))
		# this is much faster than using TensorOperations.trace!
   		@diagonal_op M iA s += M.data[iA]
   		return s
	else
		(ldims, _) = findlspaces(M, Val(LS))
		(rdims, _) = findrspaces(M, Val(RS))
		return trace_array(M.data, Val(()), Val(ldims), Val(rdims))
		# (ldims, uldims) = findlspaces(M, Val(LS))
		# (rdims, urdims) = findrspaces(M, Val(RS))
		# return tensorscalar(tensortrace(((),()), M.data, (ldims, rdims), :N))
	end
end


# Partial trace. Using a keyward would be preferable, but if we did
# it would overwrite the method that doesn't have the keyword argument.
tr(M::Tensor, space::Integer) = tr(M, (space,))
tr(M::Tensor, spaces::Dims) = tr(M, Val(spaces))
function tr(M::Tensor, ::Val{tspaces}) where {tspaces}
	LS = lspaces_int(M)
	RS = rspaces_int(M)
	S = binteger(SpacesInt, Val(tspaces))


	TLS = LS & S
	TRS = RS & S
	 
	TLS == TRS || error("Invalid spaces to be traced")
	
	(ldims, uldims) = findlspaces(M, Val(TLS))
	(rdims, urdims) = findrspaces(M, Val(TRS))

	# println(spaces)
	# println(ldims)
	# println(rdims)
	# println(uldims)
	# println(urdims)
	data_ = trace_array(M.data, Val((uldims..., urdims...)), Val(ldims), Val(rdims))
	lspaces_ = spaces(M)[uldims]
	rspaces_ = spaces(M)[urdims]
	Tensor{lspaces_, rspaces_}(data_; validate = false)
end



# helper functions
numberedsymbol(sym::Symbol, n) = Symbol(sym,'_',n)
function symtuple(sym, N)
	elems = ntuple(i -> numberedsymbol(sym, i), N)
end
function symtuple(sym, a, b)
	elems = ntuple(i -> numberedsymbol(sym, i+a-1), b-a+1)
end

@generated function trace_array(array, ::Val{udims}, ::Val{tdims1}, ::Val{tdims2}) where {udims, tdims1, tdims2}
	length(tdims1) == length(tdims2) || error("Dimensions to be traced must come in pairs")
	U = length(udims)
	T = length(tdims1)
	p = (udims..., tdims1..., tdims2...)
	isperm(p) || error("tdims1, tdims2, and udims must form a permutation")
	invp = invperm(p)

	# determine a not-terrible loop order
	cost = (ntuple(k-> max(k, udims[k]), U)..., ntuple(k->max(tdims1[k],tdims2[k]), T)...)
	looporder = sortperm(cost)

	# create symbolic index expressions
	# i_1,...,i_U index the untraced dimensions (udims)
	# i_(U+1),...,i_(U+T) index the traced dimensions (tdims1 and tdims2)
	I = symtuple(:i, 1, U)	 # indices for the result
	J = (I..., symtuple(:i, U+1, U+T)..., symtuple(:i, U+1, U+T)... )[invp]	# indices for the array

	# the loop body
	loopexpr = quote
		@inbounds R[$(I...)] += array[$(J...)]
	end

	# construct nested for loops around the body
	for loop = 1:(T+U)
		loopindex = looporder[loop]
		loopvar = numberedsymbol(:i, loopindex)
        rng = :(axes(array, $(p[loopindex])))
		loopexpr = quote
            for $loopvar in $rng
                $loopexpr
            end
        end
	end
	
	# construct the preamble and insert the for loops
	quote
		axes(array, tdims1) == axes(array, tdims2) || error("Dimensions to be traced have different axes")
		R = similar(array, size(array)[udims])
		fill!(R, zero(eltype(array)))
		$loopexpr
		return R
	end
end


# Implementation using TensorOperations (slower)
function tr_(M::Tensor)
	LS = lspaces_int(M)
	RS = rspaces_int(M)
	LS == RS || error("To trace a tensor, it must have matching left and right spaces")
	
	(ldims, _) = findlspaces(M, Val(LS))
	(rdims, _) = findrspaces(M, Val(RS))
	return tensorscalar(tensortrace(((),()), M.data, (ldims, rdims), :N))
end

tr_(M::Tensor, space::Integer) = tr_(M, (space,))
tr_(M::Tensor, spaces::Dims) = tr_(M, Val(spaces))
function tr_(M::Tensor, ::Val{tspaces}) where {tspaces}
	LS = lspaces_int(M)
	RS = rspaces_int(M)
	S = binteger(SpacesInt, Val(tspaces))


	TLS = LS & S
	TRS = RS & S
	 
	TLS == TRS || error("Invalid spaces to be traced")
	
	TS = TLS
	(ldims, uldims) = findlspaces(M, Val(TS))
	(rdims, urdims) = findrspaces(M, Val(TS))

	data_ = tensortrace((uldims, urdims), M.data, (ldims, rdims), :N)
	lspaces_ = spaces(M)[uldims]
	rspaces_ = spaces(M)[urdims]
	Tensor{lspaces_, rspaces_}(data_; validate = false)
end



# """
# 	marginal(T::Tensor, spaces)

# Trace out all but the specified spaces.
# """
# marginal(M::Tensor, space::Integer) = marginal(M, (space,))
# marginal(M::Tensor, spaces::Dims) = marginal(M, Val(spaces))

# function marginal(M::Tensor{LS,RS}, ::Val{kspaces}) where {LS,RS,kspaces}
# 	KS = binteger(SpacesInt, Val(kspaces))
# 	TSL = ~KS & LS
# 	TSR = ~KS & RS
# 	TSL == TSR || error("trace would act upon unequal left and right spaces")
# 	return tr(M, Val(TSL))
# end





#
# Operations with UniformScaling
#

# Can only add or subtract I if the Tensor is square
# function +(M::Tensor, II::UniformScaling)
# 	ensure_square(M)
# 	R = LinearAlgebra.copy_oftype(M.data, promote_type(eltype(II), eltype(M)))
# 	@diagonal_op M iR R[iR] += II.λ
# 	return Tensor(R, M)
# end
# (+)(II::UniformScaling, M::Tensor) = M + II

# function -(M::Tensor, II::UniformScaling)
# 	ensure_square(M)
# 	R = LinearAlgebra.copy_oftype(M.data, promote_type(eltype(M), eltype(II)))
# 	@diagonal_op M iR R[iR] -= II.λ
# 	return Tensor(R, M)
# end

# # NOTE:  This assumes x - A = x + (-A).
# function -(II::UniformScaling, M::Tensor)
# 	ensure_square(M)
# 	R = LinearAlgebra.copy_oftype(-M.data, promote_type(eltype(II), eltype(M)))
# 	@diagonal_op M iR R[iR] += II.λ
# 	return Tensor(R, M)
# end

# *(M::Tensor, II::UniformScaling) = M * II.λ
# *(II::UniformScaling, M::Tensor) = II.λ * M



#
# Arithmetic operations
#


# Addition and subtraction

-(M::Tensor) = Tensor(-M.data, M)

# fallback (for unequal spaces)
+(A::Tensor, B::Tensor) = error("Can only add or subtract Tensors with the same spaces")
-(A::Tensor, B::Tensor) = error("Can only add or subtract Tensors with the same spaces")


# function +(A::Tensor{LS,RS}, B::Tensor{LS,RS}) where {LS,RS}
# 	if A.ldims == B.ldims && A.rdims == B.rdims
# 		# Same spaces in the same order
# 		return Tensor(A.data + B.data, A)
# 	else
# 		# Same spaces in different order
# 		# Same code as in ==. Should we make a macro?
# 		Adims = (A.ldims..., A.rdims...)
# 		Bdims = (B.ldims..., B.rdims...)

# 		BinA = Bdims[invperm(Adims)]

# 		Rdata = A.data + permutedims(B.data, BinA)
# 		return Tensor(Rdata, A)
# 	end
# end


# function -(A::Tensor{LS,RS}, B::Tensor{LS,RS}) where {LS,RS}
# 	if A.ldims == B.ldims && A.rdims == B.rdims
# 		# Same spaces in the same order
# 		return Tensor(A.data - B.data, A)
# 	else
# 		# Same spaces in different order
# 		# Same code as in ==. Should we make a macro?
# 		Adims = (A.ldims..., A.rdims...)
# 		Bdims = (B.ldims..., B.rdims...)

# 		BinA = Bdims[invperm(Adims)]

# 		Rdata = A.data - permutedims(B.data, BinA)
# 		return Tensor(Rdata, A)
# 	end
# end

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

# """
# `T*X` where `T` is a Tensor and `X` is an `AbstractArray` contracts the right
# dimensions of `T` with dimensions `rspaces(T)` of `X`.  The result is an array of type
# similar to `X`, whose size along the contracted dimensions is `lsize(T)` and whose size in
# the uncontracted dimensions is that of `X`.

# `X*T` is similar, except the left dimensions of `T` are contracted against `X`, and the
# size of the result depends on the `rsize(M)`.
# """
# *(M::Tensor, A::SupportedArray{TA,1}) where {TA} = _mult_MA(M, A)
# *(M::Tensor, A::SupportedArray{TA,2}) where {TA} = _mult_MA(M, A)
# *(M::Tensor, A::SupportedArray{TA}) where {TA} = _mult_MA(M, A)

# function _mult_MA(M::Tensor, A::SupportedArray)
# 	nl = nlspaces(M)
# 	nr = nrspaces(M)
# 	S = rspaces(M)
# 	raxes(M) == axes(A, S) || throw(DimensionMismatch("raxes(M) must equal axes(B, rspaces(A))"))

# 	nR = max(ndims(A), maximum(S))
# 	odimsA = deleteat(oneto(nR), S)

# 	szR = MVector(size(A, oneto(nR)))
# 	lszM = lsize(M)
# 	for i = 1:nl
# 		szR[S[i]] = lszM[i]
# 	end
# 	TR = promote_type(eltype(M), eltype(A))
# 	R = similar(A, TR, Tuple(szR))
# 	tensorcontract!(one(eltype(M)), M.data, :N, A, :N, zero(TR), R, oneto(nl), tupseq(nl+1,nl+nr), odimsA, S, invperm((S...,odimsA...)))
# 	return R
# end
# #*(M::Tensor, A::SupportedArray{TA,1}) where {TA} =


# *(A::SupportedArray{TA,1}, M::Tensor) where {TA} = _mult_AM(A, M)
# *(A::SupportedArray{TA,2}, M::Tensor) where {TA} = _mult_AM(A, M)
# *(A::SupportedArray{TA}, M::Tensor) where {TA} = _mult_AM(A, M)

# # This is almost identical to the M*A version.
# function _mult_AM(A::SupportedArray, M::Tensor)
# 	error("The desired behavior of Array * Tensor has not been decided.")

# 	n = nspaces(M)
# 	S = spaces(M)
# 	laxes(M) == axes(A, S) || throw(DimensionMismatch("axes(A, spaces(B)) must equal laxes(B)"))

# 	nR = max(ndims(A), maximum(S))
# 	kdimsA = deleteat(oneto(nR), S)

# 	szR = MVector(size(A, oneto(nR)))
# 	rszM = rsize(M)
# 	for i = 1:n
# 		szR[S[i]] = rszM[i]
# 	end
# 	TR = promote_type(eltype(M), eltype(A))
# 	R = similar(A, TR, Tuple(szR))
# 	tensorcontract!(one(eltype(M)), M.data, :N, A, :N, zero(TR), R, tupseq(n+1,2*n), oneto(n), kdimsA, S, invperm((S...,kdimsA...)))
# 	return R
# end



"""
Tensor-Tensor multiplication.

	C = A*B

contracts the right spaces of Tensor `A` with matching left spaces of Tensor `B`.
The left spaces of `C` are the left spaces of `A` and the uncontracted left spaces of `B`,
provided these are distinct. (An error is thrown if they are not.)  Similarly, the right
spaces of `C` are the right spaces of `B` and the uncontracted right spaces of `A`, provided
these are distinct.  The order of the output spaces is undefined and should not be relied upon.
"""
function *(A::Tensor, B::Tensor)
	LSA = lspaces_int(A)
	RSA = rspaces_int(A)
	LSB = lspaces_int(B)
	RSB = rspaces_int(B)
	
	# Check for compatibility of spaces
	CS = RSA & LSB			# spaces to be contracted
	URSA = RSA & (~CS)		# uncontracted right spaces of A
	ULSB = LSB & (~CS)		# uncontracted left spaces of B
	
	LSA & ULSB == SpacesInt(0) || error("A and B have uncontracted left spaces in common")
	RSB & URSA == SpacesInt(0) || error("A and B have uncontracted right spaces in common")
	
	# match up indices for inner and outer products
	(cdimsA, urdimsA) = findrspaces(A, Val(CS))
	(cdimsB, uldimsB) = findlspaces(B, Val(CS))
	
	odimsA = (oneto(nlspaces(A))..., urdimsA...)
	odimsB = (uldimsB..., tuplerange(nlspaces(B)+1, ndims(B))...)
	
	lspaces_ = (lspaces(A)..., spaces(B)[uldimsB]...) 
	rspaces_ = (spaces(A)[urdimsA]..., rspaces(B)...)
	
	# Order produced by tensorcontract:  (ldimsA, urdimsA, uldimsB, rdimsB)
	blocksizes = (nlspaces(A), length(urdimsA), length(uldimsB), nrspaces(B))
	pc1 = blockperm(blocksizes, (1,3))
	pc2 = blockperm(blocksizes, (2,4))
	
	# println("cdimsA = ", cdimsA)
	# println("urdimsA = ", urdimsA)
	# println("odimsA = ", odimsA)
	# println("cdimsB = ", cdimsB)
	# println("uldimsB = ", uldimsB)
	# println("odimsB = ", odimsB)
	# println("pc1 = ", pc1)
	# println("pc2 = ", pc2)
	
	data_ = tensorcontract((pc1, pc2), A.data, (odimsA, cdimsA), :N, B.data, (cdimsB, odimsB), :N)
	return Tensor{lspaces_, rspaces_}(data_)
end


# faster version when right spaces of A exactly match the left spaces of B
function *(A::Tensor{lspacesA,cspaces}, B::Tensor{cspaces,rspacesB}) where {lspacesA, rspacesB, cspaces}
	# The right spaces of A are the same as the left spaces of B
	data_= reshape(Matrix(A) * Matrix(B), (lsize(A)..., rsize(B)...))
	Tensor{lspacesA, rspacesB}(data_)
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



#---------------------------
# Dimension-foo

function isin(spaces::Dims, ::Val{S}) where {S}
	mask = ntuple(i -> (S & (SpacesInt(1) << (spaces[i]-1))) != SpacesInt(0), length(spaces))
end


function findlspaces(M::Tensor, ::Val{S}) where {S}
	MS = binteger(SpacesInt, Val(lspaces(M)))
	sdims = lperm(M)[findnzbits(S, MS)]
	odims =oneto(nlspaces(M))[isin(lspaces(M), Val(MS & ~S))]
	# odims = findnzbits((~S & MS), MS)	# WRONG ORDER!
	return (sdims, odims)
end

function findrspaces(M::Tensor, ::Val{S})  where {S}
	MS = binteger(SpacesInt, Val(rspaces(M)))
	sdims = nlspaces(M) .+ rperm(M)[findnzbits(S, MS)]
	odims = nlspaces(M) .+ oneto(nrspaces(M))[isin(rspaces(M), Val(MS & ~S))]
	# odims = nlspaces(M) .+ findnzbits((~S & MS), MS)	# WRONG ORDER!
	return (sdims, odims)
end


function blockperm(siz, perm)
	cumsiz = (0, cumsum(siz[1:end-1])...)
	_blockperm(siz[perm], cumsiz[perm])
end


_blockperm(siz::Dims{1}, cumsiz::Dims{1}) = ntuple(i -> cumsiz[1] + i, siz[1])
function _blockperm(siz::Dims, cumsiz::Dims) 
	thisperm = ntuple(i -> cumsiz[1] + i, siz[1])
	(thisperm..., _blockperm(tail(siz), tail(cumsiz))...)
end


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




end
