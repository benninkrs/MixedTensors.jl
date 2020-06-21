"""
A MultiMatrix is a tensor with an equal number of covarient ("left") and
contravarient ("right") indices.


"""

module MultiMatrices

export MultiMatrix, lsize, rsize, spaces, nspaces, outprod, ⊗, tr_dims

import Base: ndims, length, size
import Base: reshape, permutedims, adjoint, transpose, Matrix
import Base: getindex, setindex!
import Base: display, summary, array_summary
import Base.(*)

using BaseExtensions
using StaticArrays
using TensorOperations: trace!, contract!
using TypeTools: asdeclared
import LinearAlgebra.tr

# T element type
# S tuple of spaces
# A <: AbstractArray{T,2*S}
# Construction:
# 	MultiMatrix(array, spaces)
# 	MultiMatrix(array)	takes spaces to be 1:ndims(A)/2
# If M is a multimatrix, M(5,3) assigns it to spaces (5,3)
struct MultiMatrix{T, S, A<:AbstractArray, N} <: AbstractArray{T, N}
	data::A
	function MultiMatrix(arr::A, S::Dims{M}) where {M} where A<:AbstractArray{T,N} where {T,N}
		if N != 2*M
			error("ndims(A) must be twice the number of specified spaces.")
		else
			return new{T,S,A,2*M}(arr)
		end
	end
end


function MultiMatrix(arr::A) where A<:AbstractArray{T,N} where {T,N}
	iseven(N) || error("Source array must have an even number of dimensions")
	MultiMatrix(arr, oneto(N>>1))
end


# change dimensions
(m::MultiMatrix)(spaces::Vararg{Int64}) = m(spaces)
(m::MultiMatrix)(spaces::Tuple{Vararg{Int64}}) = MultiMatrix(m.data, spaces)


# Construct a MultiMatrix out of a vector
#multivector(v) = MultiMatrix(reshape(v, [size(v); ones(ndims(v))]))

#summary(io::IO, m::MultiMatrix) = array_summary(io, m, axes(m.data))

spaces(m::MultiMatrix{T,S}) where {T,S} = S
nspaces(m::MultiMatrix{T,S}) where {T,S} = length(S)

# Size and shape
ndims(m::MultiMatrix) = ndims(m.data)
length(m::MultiMatrix) = length(m.data)
size(m::MultiMatrix) = size(m.data)
size(m::MultiMatrix, dims) = size(m.data, dims)
lsize(m::MultiMatrix) =  ntuple(d -> size(m.data, d), nspaces(m))
rsize(m::MultiMatrix) = ntuple(d -> size(m.data, d+nspaces(m)), nspaces(m))


# Conversions
convert(t, mm::MultiMatrix) = convert(t, mm.data)


"""
Return a `Matrix` obtained by reshaping a `MultiMatrix`. The first `n`
dimensions are combined into the first dimension of the matrix, while the last
`n` dimensions are combined into the second dimension of the matrix.
"""
Matrix(m::MultiMatrix) = reshape(m.data, prod(lsize(m)), prod(rsize(m)) )


# If accessing a single element, return that element.
# Otherwise (i.e. if requesting a range) return a MultiMatrix.

getindex(m::MultiMatrix, i::CartesianIndex) = getindex(m, Tuple(i))

function getindex(m::MultiMatrix{T,S}, i::Vararg{Integer}) where {T,S}
	# extract

	ni = length(i)
	iseven(ni) || error("must index with an even number of indices")
	nspc = ni >> 1
	ileft = MVector{nspc}(i[1:nspc])
	iright = MVector{nspc}(i[nspc+1:2*nspc])

	for s in S
		ileft[s] = 0
		iright[s] = 0
	end

	show(ileft)
	show(iright)
	show(i[S])
	show(i[S+nspc])

	if ileft == iright
		return m.data[i[S]..., i[S+nspc]...]
	else
		return zero(T)
	end

end

#getindex(m::MultiMatrix, i::CartesianIndex) = getindex(m.data, i)
#getindex(m::MultiMatrix, i::Vararg{Integer}) = getindex(m.data, i...)
#getindex(m::MultiMatrix, i...) = MultiMatrix(getindex(m.data, i...))
#setindex!(m::MultiMatrix, i...) = setindex!(m.data, i...)



reshape(m::MultiMatrix, shape::Dims) = MultiMatrix(reshape(m.data, shape))

permutedims(m::MultiMatrix, ord) = MultiMatrix(permutedims(m.data, ord))

# Not needed.  Use M(spaces(M)[p]) if you really want a permutation.
#
# # Permute spaces (pairs of dimensions).
# function permute_spaces(m::MultiMatrix, p)
# 	return MultiMatrix(permutedims(m.data), [p; p+nspaces(m)])
# end


#findspace(m::MultiMatrix{T,S}, s::Integer) where {T,S} = findfirst(s .== S)

#Swap left and right dimensions of a MultiMatrix
function transpose(m::MultiMatrix)
	n = nspaces(m)
	return MultiMatrix(permutedims(m.data, [n+1:2*n; 1:n]))
end

# Partial transpose
function transpose(m::MultiMatrix{T,S}, spaces::Dims) where {T,S}
	n = nspaces(m)
	dims = map(s -> findfirst(s .== S), spaces)
	order = MVector{n}(1:n)	# TODO: WRONG!  Should be 2n
	for d in dims
		order[d] = d+n
		order[d+n] = d
	end
	return MultiMatrix(permutedims(m.data, order))
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

# Partial trace
tr(arr::MultiMatrix, s::Integer) where {T,S} = tr(arr, (s,))


function tr(arr::MultiMatrix{T,S,A,N}, tspaces::Dims{M}) where {T,S,A,N,M}
	nspc = length(S)

	smap = falses(maximum(S),1)
	for t in tspaces
		if t>0 && t<=maximum(S)
			smap[t] = true
		end
	end

	#(tdims, kdims) = findtruefalse(mask)
	#tdims = Tuple(filter(i->smap[S[i]], 1:nspc))
	#kdims = Tuple(filter(i->!smap[S[i]], 1:nspc))

	#D1 = Vector{Int64}(undef, nspc)
	#D2 = Vector{Int64}(undef, nspc)
	D1 = MVector{nspc,Int64}(ntup(0,nspc))
	D2 = MVector{nspc,Int64}(ntup(0,nspc))
	i1 = 0
	i2 = 0
	for d in 1:nspc
		if smap[S[d]]
			i1 += 1
			D1[i1] = d
		else
			i2 += 1
			D2[i2] = d
		end
	end

	D1tup = Tuple(D1)
	D2tup = Tuple(D2)
	tdims = Tuple(D1)[oneto(i1)]
	kdims = Tuple(D2)[oneto(i2)]

	# For some reason it's slower when I use this instead, which has the exact same code in it
	#(tdims, kdims) = get_tr_dims(arr, tspaces)
	return tr_dims(arr, tdims, kdims) #, tdims, kdims, tdims_, kdims_
end


function get_tr_dims(arr::MultiMatrix{T,S,A,N}, tspaces::Dims{M}) where {T,S,A,N,M}
	nspc = length(S)

	smap = falses(maximum(S),1)
	for t in tspaces
		if t>0 && t<=maximum(S)
			smap[t] = true
		end
	end

	#(tdims, kdims) = findtruefalse(mask)
	#tdims = Tuple(filter(i->smap[S[i]], 1:nspc))
	#kdims = Tuple(filter(i->!smap[S[i]], 1:nspc))

    #D1 = Vector{Int64}(undef, nspc)
	#D2 = Vector{Int64}(undef, nspc)
	D1 = MVector{nspc,Int64}(ntup(0,nspc))
	D2 = MVector{nspc,Int64}(ntup(0,nspc))
    i1 = 0
	i2 = 0
    for d in 1:nspc
        if smap[S[d]]
			i1 += 1
            D1[i1] = d
        else
			i2 += 1
			D2[i2] = d
		end
    end

	tdims = Tuple(D1)[oneto(i1)]
	kdims = Tuple(D2)[oneto(i2)]

	return (tdims, kdims)
end



"""
`reduced(A, spaces)` returns the reduction of `A` to the specified spaces, by
tracing out all other spaces.  (The result is always a `MultiMatrix`, even if
`spaces` is empty.)
"""
#reduced()


function tr_dims(arr::MultiMatrix{T,S,A,N}, tdims, kdims::Dims{K}) where {T,S,A,N,K}
	nspc = length(S)
	lsz = lsize(arr)
	rsz = rsize(arr)
	R = asdeclared(A){T,2*K}(undef, lsz[kdims]..., rsz[kdims]...)
	trace!(1, arr.data, :N, 0, R, kdims, kdims+nspc, tdims, tdims+nspc)
	return MultiMatrix(R, S[kdims])
end



# function tr_notbad(arr::MultiMatrix{T,S,A,N}, tspaces::Dims{M}) where {T,S,A,N,M}
# 	nspc = length(S)
# 	mask = falses(length(S))
#
# 	# Map traced spaces to array dimensions
# 	for t in tspaces
# 		r = findfirst(t .== S)
# 		if !isnothing(r)
# 			mask[r] = true
# 		end
# 	end
# 	(tdims, kdims) = findtruefalse(mask)
#
# 	return tr_dims(arr, tdims, kdims)
# end




"""
EmbeddedMultiMatrix

A multimatrix that acts on particular spaces (not just 1:nspaces).
Basic operations on embedded multimatrices (indexing, size, ...) refer to the
dimensions of the underlying data.  The spaces only come into play when performing
algebraic operations (+,-,*,trace)
"""

function (*)(A::EmbeddedMultiMatrix, B::AbstractArray)
	# Contrct spaces of A with corresponding dims of B.
	#
end

function (*)(A::EmbeddedMultiMatrix, B::EmbeddedMultiMatrix)
	# Let SA, SB be spaces of A,B.
	# If SA ∩ SB = 0, then form a lazy outer product.
	# Otherwise,
	if length(SA) <= length(SB)
		# There's got to be a more consolidated way to do this
		spc_in_SB = findin(SA, SB)
		SA_paired = spc_in_SB .> 0
		ldimsA = antiselect(oneto(length(SA), SA_paired)
		rdimsA = select(oneto(length(SA)), SA_paired)
		ldimsB = select(spc_in_SB, SA_paired)
		rdimsB = antiselect(oneto(length(SB)), spc_in_SB[SA_paired])
	end
end






end
