using AlgebraicTensors
using BenchmarkTools
using LinearAlgebra: I, tr
using Test




A = Tensor(reshape(1.0:24, (1,2,3,4)), (1,2));
B = Tensor(reshape(1.0:24, (3,4,1,2)), 1:2);
C = Tensor(reshape([1222.0, 1300.0, 2950.0, 3172.0], (1,2,1,2)), 1:2);
R = Tensor(rand(3,4,3,4), 1:2);

AA = Tensor(rand(4,3,2,1,6,5,4,3), 1:4, 5:8);
BB = Tensor(rand(6,5,4,3,5,4,3,2), 5:8, 9:12);

@info "Testing (re)construction"
A_ = A(3,2);
@test spaces(A_) == ((3,2), (3,2))
@btime A_ = ($A)(3,2);

A_ = A((5,6),(8,7));
@test spaces(A_) == ((5,6), (8,7))
@btime A_ = ($A)((5,6),(8,7));


@info "testing getindex"
T = Tensor(randn(2,3,4,5,6), (5,2,3), (10,60));
S = T[:,2,:,4,:];
@test spaces(S) == ((5,3), (60,))
@btime ($T)[:,2,:,4,:];
@btime AlgebraicTensors.getindex_($T, (:,2,:,4,:));


@info "testing transpose"
T = Tensor(randn(2,3,4,5,6), (2,5,3), (1,5));
S = transpose(T, 5);
@test spaces(S) == ((2,3,5), (1,5))
@test size(S) == (2,4,6,5,3)
@btime transpose($T, 5);		# 

S = transpose(T, (5,2));
@test spaces(S) == ((3,5), (1,2,5))
@test size(S) == (4,6,5,2,3)
@btime transpose($T, (5,2));	# 890 nns


@info "testing multiplication"
@test A*B == C
@test_throws DimensionMismatch A*B(2,1)
@test A(9,5)*B(9,5) == C(9,5)

CC = AA*BB;
@test spaces(CC) == ((1,2,3,4), (9,10,11,12))

@info "Benchmarking small multiplication."  #  Expect 300 ns (7 allocations: 416 bytes)
@btime $A*$B;

@info "Benchmarking large multiplication."    #  Expect 116 μs (2 allocations: 23 KiB)
@btime $AA*$BB;


@info "Benchmarking small trace."  # Expect 19 ns (2 allocations: 96 bytes)
@btime tr($R);

@info "Testing small partial trace."	# Expect 473 ns (12 allocations: 816 bytes)
@btime tr($R, 2);


@info "Combination tests"
@test size(A + B') == (1,2,3,4)
@test R * inv(R) ≈ Tensor(reshape(I(12), (3,4,3,4)), 1:2)


# tr(Matrix(R)) takes only 14 ns.
