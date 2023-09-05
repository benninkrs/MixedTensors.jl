"""
	AlgebraicTensors (module)

Implements tensors of arbitrary dimensionality and size as algebraic objects.
"""

#=
8/20/2023
	This is a major revision to (1) make getindex type stable and (2) use the new TensorOperation.jl API.
	I am going to return to the original idea parameterizing the type with tuples for the ordered left and
	right spaces. Hopefully compiler improvements have made this inferrable now.
	This should make all typed functions fast, but may result in excessive code generation.
=#

#=
- In A+B and A-B, the order of spaces is that of A
- In A*B, the left spaces of A precede the uncontracted left spaces of B,
	followed by the uncontracted right spaces of A, then right spaces of B.
	When applying a tensor operator to a tensor vector, the order of spaces
	in the vector will be changed.  This is sort of undesirable ... ?
=#


#=
COMPLETED:
Constructors			Done, performant
getindex					Done, performant
setindex!				Done, performant
==							Done, performant
+,-						Done, performant (but can be better)
adjoint					Done, performant
transpose				Done, performant
partial transpose		Done, performant
Tensor*Tensor			Done, performant
trace						Done, performant
partial trace			Done, performant
analytic funcs			Done, performant
det						Done, performant
opnorm					Done, performant
eigvals					Done, performant
svdvals					Done, performant
eig						Done, performant
svd						Dome, performant

ISSUES:
- eig, ^, and other square operator operations currently give wrong answers if lspaces,
 rspaces have different orderings

TODO:
- transpose, adjoint
- Implement IndexLinear versions for +,-,== (will be faster)
- Tensor*Array?
- broadcasting
- lazy tensor products
- better mechanism for comparison, element-wise ops?
- test performance in non-inferrable cases
=#


module AlgebraicTensors

export Tensor, lsize, rsize, spaces, lspaces, rspaces, nlspaces, nrspaces
export tr, marginal, det, opnorm, eigvals, eigvecs, svdvals, svd, opnorm

export tensor_op

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
import LinearAlgebra: tr, det, eigvals, svdvals, opnorm, eigvecs, svd



#------------------------------------
# Definitions

const SpacesInt = UInt128  		# An integer treated as a bit set for vector spaces
const MaxSpace = 	128				# number of bits in SpacesInt (maximum space index)
const Spaces{N} = Tuple{Vararg{Integer,N}}
const Iterable = Union{Tuple, AbstractArray, UnitRange, Base.Generator}
const Axes{N} = NTuple{N, AbstractUnitRange{<:Integer}}
const SupportedArray{T,N} = DenseArray{T,N}		# can change this later.  Maybe to StridedArray?


# # Can we find a better approach to calc_strides and diagonal_op?
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

			all(LS .> 0) && all(LS .<= MaxSpace) || error("Values of LSPACES must be integer between 1 and $nbits")
			all(RS .> 0) && all(RS .<= MaxSpace) || error("Values of RSPACES must be integer between 1 and $nbits")
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
end



# Convenience constructors
Tensor{LS}(arr) where {LS} = Tensor{LS, LS}(arr)
Tensor(arr) = Tensor{oneto(ndims(arr)), ()}(arr)

# Construct from an array, using another Tensor's metadat.
	# # Shortcut constructor: construct from array, using another Tensor's metadata.
	# # By ensuring the array has the right number of dimensions, no input checking is needed.
	
function Tensor(arr, M::Tensor{LS,RS}) where {LS,RS}
	ndims(arr) == ndims(M) || error("Array must have the same number of dimensions as the template tensor")
	Tensor{LS,RS}(arr; validate = false)
end


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

# return the sort order of the spaces.
# This can be viewed as a map from ordered spaces to axes (with an offset for rperm)
# lperm(M::Tensor) = sortperm(lspaces(M))
# rperm(M::Tensor) = sortperm(rspaces(M))

# the dimensions of the sorted left and right spaces 
lspace_dims(M::Tensor) = sortperm(lspaces(M))
rspace_dims(M::Tensor) = nlspaces(M) .+ sortperm(rspaces(M))

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



#-----------------------------------------
# Comparison

# Fast comparison if X,Y have same spaces in the same order
==(X::Tensor{LS,RS}, Y::Tensor{LS,RS}) where {LS,RS} = (X.data == Y.data)

function ==(X::Tensor, Y::Tensor)

	LX = lspaces_int(X)
	RX = rspaces_int(X)
	LY = lspaces_int(Y)
	RY = rspaces_int(Y)

	# Result is false if X,Y have different spaces
	if (LX != LY) || (RX != RY)
		return false
	end

	# Same spaces but different order

	ldimsX = lspace_dims(X)
	rdimsX = rspace_dims(X)

	ldimsY = lspace_dims(Y)
	rdimsY = rspace_dims(Y)

	# check that corresponding axes in X,Y are the same
	if !(axes(X.data, ldimsX) == axes(Y.data, ldimsY) && axes(X.data, rdimsX) == axes(Y.data, rdimsY))
		return false
	end

	test_array_equality(X.data, Y.data, Val(ldimsX), Val(rdimsX), Val(ldimsY), Val(rdimsY))
end



# No checking here.  Assumes X,Y have same spaces with the same sizes, but possibly in different order
@generated function test_array_equality(X, Y, ::Val{ldimsX}, ::Val{rdimsX}, ::Val{ldimsY}, ::Val{rdimsY}) where {ldimsX, rdimsX, ldimsY, rdimsY}
	NL = length(ldimsX)
	NR = length(rdimsX)
	Xdims = (ldimsX..., rdimsX...) 
	Ydims = (ldimsY..., rdimsY...) 
	Xperm = invperm(Xdims)
	Yperm = invperm(Ydims)

	# determine a not-terrible loop order
	cost = (ntuple(k->max(ldimsX[k], ldimsY[k]), NL)..., ntuple(k->max(rdimsX[k], rdimsY[k]), NR)...)
	looporder = sortperm(cost)

	# create symbolic index expressions
	# i_1,...,i_U index the untraced dimensions (udims)
	# i_(U+1),...,i_(U+T) index the traced dimensions (tdims1 and tdims2)
	I = symtuple(:i, 1, NL+NR)
	IX = I[Xperm]
	IY = I[Yperm]

	# the loop body
	loopexpr = quote
		@inbounds if X[$(IX...)] !== Y[$(IY...)]
			return false
		 end
	end

	# construct nested for loops around the body
	for loop = 1:(NL+NR)
		loopindex = looporder[loop]
		loopvar = Symbol(:i, '_', loopindex)
        rng = :(axes(X, $(Xdims[loopindex])))
		loopexpr = quote
            for $loopvar in $rng
                $loopexpr
            end
        end
	end
	
	# construct the preamble and insert the for loops
	quote
		$loopexpr
		return true
	end
end




# # Version where the index permutation is determined at run-time.
# # This is much slower.

# function isequal_(X::Tensor, Y::Tensor)

# 	LX = lspaces_int(X)
# 	RX = rspaces_int(X)
# 	LY = lspaces_int(Y)
# 	RY = rspaces_int(Y)

# 	# Result is false if X,Y have different spaces
# 	if (LX != LY) || (RX != RY)
# 		return false
# 	end

# 	# Same spaces but different order

# 	ldimsX =lspace_dims(X)
# 	rdimsX =rspace_dims(X)

# 	ldimsY =lspace_dims(Y)
# 	rdimsY =rspace_dims(Y)

# 	# check that corresponding axes in X,Y are the same
# 	if !(axes(X.data, ldimsX) == axes(Y.data, ldimsY) && axes(X.data, rdimsX) == axes(Y.data, rdimsY))
# 		return false
# 	end

# 	test_array_equality_(X.data, Y.data, ldimsX, rdimsX, ldimsY, rdimsY)
# end

# @generated function test_array_equality_(X, Y, ldimsX::Dims{NL}, rdimsX::Dims{NR}, ldimsY::Dims{NL}, rdimsY::Dims{NR}) where {NL,NR}


# 	# create symbolic index expressions
# 	# i_1,...,i_U index the untraced dimensions (udims)
# 	# i_(U+1),...,i_(U+T) index the traced dimensions (tdims1 and tdims2)
# 	I = Expr(:tuple, symtuple(:i, 1, NL+NR)...)

# 	# the loop body
# 	loopexpr = quote
# 		I = $I
# 		IX = I[Xperm]
# 		IY = I[Yperm]
# 		# @inbounds
# 			if X[IX...] !== Y[IY...]
# 			return false
# 		 end
# 	end

# 	# construct nested for loops around the body
# 	for loop = 1:(NL+NR)
# 		loopvar = numberedsymbol(:i, loop)
#         rng = :(axes(X, Xdims[$loop]))
# 		loopexpr = quote
#             for $loopvar in $rng
#                 $loopexpr
#             end
#         end
# 	end
	
# 	# construct the preamble and insert the for loops
# 	quote
# 		Xdims = (ldimsX..., rdimsX...) 
# 		Ydims = (ldimsY..., rdimsY...) 
# 		Xperm = invperm(Xdims)
# 		Yperm = invperm(Ydims)
	
# 		$loopexpr
# 		return true
# 	end
# end



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




# Partial transpose
"""
	transpose(tensor, space)
	transpose(tensor, spaces)
Toggle the "direction" (left or right) of specified tensor spaces. Any `spaces` that are not
present in `tensor` are ignored.  Transposed spaces are placed after the other spaces.
"""
transpose(M::Tensor, ts::Int) = transpose(M, Val((ts,)))
transpose(M::Tensor, ts::Dims) = transpose(M, Val(ts))
function transpose(M::Tensor, ::Val{tspaces}) where {tspaces}
	LS = lspaces_int(M)
	RS = rspaces_int(M)
	
	NL = nlspaces(M)
	NR = nrspaces(M)

	TS = binteger(SpacesInt, Val(tspaces))
	TSL = TS & LS		# transposed spaces on the left side
	TSR = TS & RS		# transposed spaces on the right side


	if iszero(TSL) && iszero(TSR)
		# nothing to transpose
		return M
	else
		(tldims, oldims) = findlspaces(M, Val(TSL))
		(trdims, ordims) = findrspaces(M, Val(TSR))

		ldims = (oldims..., trdims...)
		rdims = (ordims..., tldims...)

		arr = permutedims(M.data,  (ldims..., rdims...))

		lspaces_ = spaces(M)[ldims];
		rspaces_ = spaces(M)[rdims];

		return Tensor{lspaces_, rspaces_}(arr)
	end
end



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
		# this is much faster than using TensorOperations.trace!
   		s = zero(eltype(M))
   		@diagonal_op M iA s += M.data[iA]
   		return s
		# return tr(Matrix(M))
		
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
tr(M::Tensor, space::Integer) = tr(M, Val((space,)))
tr(M::Tensor, spaces::Dims) = tr(M, Val(spaces))
function tr(M::Tensor, ::Val{tspaces}) where {tspaces}
	LS = lspaces_int(M)
	RS = rspaces_int(M)
	if tspaces isa SpacesInt
		S = tspaces
	elseif tspaces isa Dims
		S = binteger(SpacesInt, Val(tspaces))
	else
		error("invalid argument")
	end

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
		loopvar =Symbol(:i, '_', loopindex)
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


# # Implementation using TensorOperations (slower)
# function tr_(M::Tensor)
# 	LS = lspaces_int(M)
# 	RS = rspaces_int(M)
# 	LS == RS || error("To trace a tensor, it must have matching left and right spaces")
	
# 	(ldims, _) = findlspaces(M, Val(LS))
# 	(rdims, _) = findrspaces(M, Val(RS))
# 	return tensorscalar(tensortrace(((),()), M.data, (ldims, rdims), :N))
# end

# tr_(M::Tensor, space::Integer) = tr_(M, (space,))
# tr_(M::Tensor, spaces::Dims) = tr_(M, Val(spaces))
# function tr_(M::Tensor, ::Val{tspaces}) where {tspaces}
# 	LS = lspaces_int(M)
# 	RS = rspaces_int(M)
# 	S = binteger(SpacesInt, Val(tspaces))


# 	TLS = LS & S
# 	TRS = RS & S
	 
# 	TLS == TRS || error("Invalid spaces to be traced")
	
# 	TS = TLS
# 	(ldims, uldims) = findlspaces(M, Val(TS))
# 	(rdims, urdims) = findrspaces(M, Val(TS))

# 	data_ = tensortrace((uldims, urdims), M.data, (ldims, rdims), :N)
# 	lspaces_ = spaces(M)[uldims]
# 	rspaces_ = spaces(M)[urdims]
# 	Tensor{lspaces_, rspaces_}(data_; validate = false)
# end



"""
	marginal(T::Tensor, spaces)

Trace out all but the specified spaces.
"""
marginal(M::Tensor, space::Integer) = marginal(M, Val((space,)))
marginal(M::Tensor, spaces::Dims) = marginal(M, Val(spaces))

function marginal(M::Tensor, ::Val{kspaces}) where {kspaces}
	LS = lspaces_int(M)
	RS = rspaces_int(M)
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


function ensure_same_spaces(A::Tensor, B::Tensor)
	(lspaces_int(A) == lspaces_int(B)) && (rspaces_int(A) == rspaces_int(B)) ||
error("A,B must have the same spaces")
	nothing
end
# Addition and subtraction

-(M::Tensor) = Tensor(-M.data, M)


# fast implementation when A,B have same spaces in same order
+( A::Tensor{LS,RS}, B::Tensor{LS,RS}) where {LS,RS} = Tensor(A.data + B.data, A)
-( A::Tensor{LS,RS}, B::Tensor{LS,RS}) where {LS,RS} = Tensor(A.data - B.data, A)

# A,B have different spaces, or same spaces in different order
function +(A::Tensor, B::Tensor)
	ensure_same_spaces(A,B)

	# Same spaces in different order
	Adims = (lspace_dims(A)..., rspace_dims(A)...)
	Bdims = (lspace_dims(B)..., rspace_dims(B)...)

	BinA = Bdims[invperm(Adims)]

	return Tensor(permuted_op(+, A.data, B.data, Val(BinA)), A)
end



function -(A::Tensor, B::Tensor)
	ensure_same_spaces(A,B)

	# Same spaces in different order
	Adims = (lspace_dims(A)..., rspace_dims(A)...)
	Bdims = (lspace_dims(B)..., rspace_dims(B)...)

	BinA = Bdims[invperm(Adims)]

	return Tensor(permuted_op(-, A.data, B.data, Val(BinA)), A)
end


# this is faster than both Base.+ and Tensoroperations.tensoradd (!)
@generated function permuted_op(op, A, B, ::Val{perm}) where {perm}
	N = length(perm)

	# determine a not-terrible loop order
	cost = ntuple(k-> max(k, perm[k]), N)
	looporder = sortperm(cost)

	# create symbolic index expressions
	IA = symtuple(:i, N)
	IB = IA[perm]

	# the loop body
	loopexpr = quote
		@inbounds C[$(IA...)] = op(C[$(IA...)], B[$(IB...)])
	end

	# construct nested for loops around the body
	for loop = 1:N
		loopindex = looporder[loop]
		loopvar = Symbol(:i, '_', loopindex)
        rng = :(axes(A, $loopindex))
		loopexpr = quote
            for $loopvar in $rng
                $loopexpr
            end
        end
	end
	
	# construct the preamble and insert the for loops
	quote
		C = copy(A)
		$loopexpr
		return C
	end
end



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
	axes(A)[rspace_dims(A)] == axes(B)[lspace_dims(B)] || throw(Base.DimensionMismatch("non-matching sizes in contracted dimensions")) 
	data_= reshape(Matrix(A) * Matrix(B), (lsize(A)..., rsize(B)...))
	Tensor{lspacesA, rspacesB}(data_)
end


#-----------------------------------------------
# Linear algebraic tensor functions.
# Most of these are applicable only for "square" tensors, i.e. tensors
# that have the same left and right spaces (not necessarily in the same order)
# and have with corresponding axes the same size.
# A tensor is "proper square" if it is square and the left and right spaces are
# in the same order.

# TODO: THESE NEED TO ACCOUNT FOR THE ORDER OF SPACES!!


# Internal functions to make ensure a Tensor is square.
# ensure_square(M) throws an error if M is not square; otherwise it returns nothing
@inline function ensure_square(M::Tensor)
	ldims = lspace_dims(M)
	rdims = rspace_dims(M)
	is_square = (spaces(M)[ldims] == spaces(M)[rdims]) && (axes(M)[ldims] == axes(M)[rdims])
	is_square ? nothing : throw(DimensionMismatch("Tensor is not square"))
end


# function proper_matrix(M::Tensor)
# 	ensure_square(M::Tensor)
# 	ldims = lperm(M)
# 	rdims = rperm(M) .+ nlspaces(M)
# 	perm = (ldims..., rdims...)
# 	data_ = permutedims(M.data, perm)
# 	matrix = reshape(data_, (prod(lsize(M)), prod(rsize(M))))
# 	return (matrix, invperm(perm))
# end


# Apply a function to a square tensor
function matrix_fun(M::Tensor, f::F) where {F}
	if lspaces(M) == rspaces(M)
		laxes(M) == raxes(M) || error("Left and right spaces are different sizes")
		matrix_ = f(Matrix(M))
		arr_ = reshape(matrix_, size(M))
		return Tensor(arr_, M)
	else
		ldims = lspace_dims(M)
		rdims = rspace_dims(M)
		spaces(M)[ldims] == spaces(M)[rdims] || error("Tensor is not square")
		axes(M)[ldims] == axes(M)[rdims] || error("Left and right spaces are different sizes")
		
		perm = (ldims..., rdims...)
		sz = size(M)[perm]
		arr = permutedims(M.data, perm)
		matrix = reshape(arr, (prod(lsize(M)), prod(rsize(M))))
		matrix_ = f(matrix)
		arr_ = permutedims(reshape(matrix_, sz), invperm(perm))
		return Tensor(arr_, M)
	end
end


# Exponentiation

^(M::Tensor, p::Number) = matrix_fun(M, x -> x^p)
^(M::Tensor, p::Integer) = matrix_fun(M, x -> x^p)

# ^(M::Tensor, x::Number) = tensor_exp(M, x)
# ^(M::Tensor, x::Integer) = tensor_exp(M, x)
# function tensor_exp(M::Tensor, x)
# 	(matrix, iperm) = proper_matrix(M)
# 	Tensor(permutedims(reshape(matrix^x, size(M)), iperm), M)
# end

#
# Analytic matrix functions
#
for f in [:inv, :sqrt, :exp, :log, :sin, :cos, :tan, :sinh, :cosh, :tanh]
	@eval $f(M::Tensor) = matrix_fun(M, x -> $f(x))
	# @eval function $f(M::Tensor)
	# 		(matrix, iperm) = proper_matrix(M)
	# 		Tensor(permutedims(reshape($f(matrix), size(M)), iperm), M)
	# 	end
end


#----------- TODO

# Put a tensor's right spaces in the same order as the left spaces, then convert to a matrix
function propermatrix(M::Tensor)
	ensure_square(M);
	if lspaces(M) == rspaces(M)
		return Matrix(M)
	else
		lperm = oneto(nlspaces(M))
		rperm = rspace_dims(M)[invperm(lspace_dims(M))]
		pdata = permutedims(M.data, (lperm..., rperm...))
		return reshape(pdata, (prod(lsize(M)), prod(rsize(M))))
	end
end

#
# Linear-algebra functions
#
det(M::Tensor) = det(propermatrix(M))
opnorm(M::Tensor, args...) = opnorm(propermatrix(M), args...)

eigvals(M::Tensor, args...) = eigvals(propermatrix(M), args...)
svdvals(M::Tensor, args...) = svdvals(propermatrix(M), args...)

function eigvecs(M::Tensor, args...)
	sz = (lsize(M)..., prod(rsize(M)))
	V = reshape(eigvecs(propermatrix(M), args...), sz)
	return Tensor{lspaces(M), (MaxSpace,)}(V)
end


"""
	svd(T::Tensor)

computes the singular value decomposition of tensor `T`.  For technical reasons the result
cannot be returned as an `SVD` object, but is instead returned as a NamedTuple (:U, :S, :Vt).
`U`` is a tensor of left singular vectors, with the left spaces of `T` and right space `128`.
`Vt` is a tensor of right singular vectors, with left space `128` and the right spaces of `T`.
`S` is a vector of singular values.
"""
function svd(M::Tensor{LS}, args...) where {LS}
	ensure_square(M)
	result = svd(Matrix(M), args...)
	nsvd = length(result.S)
	U_ = Tensor{lspaces(M), (MaxSpace,)}(reshape(result.U, (lsize(M)..., nsvd)))
	Vt_ = Tensor{(MaxSpace,), rspaces(M)}(reshape(result.Vt, (nsvd, rsize(M)...)))
	return (U = U_, S = result.S, Vt = Vt_)
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

# Help functions for generated functions

# Create a tuple of sequentially numbered symbols
function symtuple(sym, N)
	elems = ntuple(i -> Symbol(sym,'_',i), N)
end

function symtuple(sym, a, b)
	elems = ntuple(i -> Symbol(sym,'_',i+a-1), b-a+1)
end

# Determine which elements of a tuple are in another.
# Returns a Bool tuple of the same size as the input
function isin(spaces::Dims, ::Val{S}) where {S}
	S isa SpacesInt || error("S must be a SpacesInt")
	mask = ntuple(i -> (S & (SpacesInt(1) << (spaces[i]-1))) != SpacesInt(0), length(spaces))
end

# find the dimensions of selected left spaces
#  sdims = dimensions corresponding to the ordered spaces specified by s
#  odims = remaining dimensions in original order
function findlspaces(M::Tensor, ::Val{S}) where {S}
	MS = binteger(SpacesInt, Val(lspaces(M)))
	sdims = lspace_dims(M)[findnzbits(S, MS)]
	odims =oneto(nlspaces(M))[isin(lspaces(M), Val(~S))]
	# odims = findnzbits((~S & MS), MS)	# WRONG ORDER!
	return (sdims, odims)
end

# find the dimensions of selected right spaces
function findrspaces(M::Tensor, ::Val{S})  where {S}
	MS = binteger(SpacesInt, Val(rspaces(M)))
	sdims = rspace_dims(M)[findnzbits(S, MS)]
	odims = nlspaces(M) .+ oneto(nrspaces(M))[isin(rspaces(M), Val(~S))]
	# odims = nlspaces(M) .+ findnzbits((~S & MS), MS)	# WRONG ORDER!
	return (sdims, odims)
end

# Construct a permutation formed by reordering blocks of given sizes
function blockperm(siz, perm)
	cumsiz = (0, cumsum(siz[1:end-1])...)
	_blockperm(siz[perm], cumsiz[perm])
end

_blockperm(siz::Dims{1}, cumsiz::Dims{1}) = ntuple(i -> cumsiz[1] + i, siz[1])
function _blockperm(siz::Dims, cumsiz::Dims) 
	thisperm = ntuple(i -> cumsiz[1] + i, siz[1])
	(thisperm..., _blockperm(tail(siz), tail(cumsiz))...)
end


# # Compute the dimensions of selected spaces after the others are contracted out
# function remain_dims(dims::Dims{N}, ::Val{S}, ::Val{K}) where {N, S, K}
# 	# S and K should be SpacesInts
# 	count_ones(S) == N || error("count_ones(S) must equal length(dims)")
# 	idims = findnzbits(K, S)
# 	kdims = dims[idims]
# 	dims_ = sortperm(kdims)		# it infers!
# 	# Should be equivalent
# 	# dims_ = ntuple(Val(N_)) do i
# 	# 	static_fn(0, Val(length(kdims))) do a,j
# 	# 		kdims[j] <= kdims[i] ? a+1 : a		
# 	# 	end
# 	# end
# end


end
