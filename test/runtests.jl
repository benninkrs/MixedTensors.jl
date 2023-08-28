using AlgebraicTensors
using BenchmarkTools
using LinearAlgebra: I, tr
using Test


# Times shown are for my HP ZBook Firefly G10, i7-1360P

A = Tensor{(1,2)}(reshape(1.0:24, (1,2,3,4)));
B = Tensor{1:2}(reshape(1.0:24, (3,4,1,2)));
C = Tensor{(1,2)}(reshape([1222.0, 1300.0, 2950.0, 3172.0], (1,2,1,2)));
R = Tensor{(1,2)}(rand(3,4,3,4));

AA = Tensor{1:4, 5:8}(rand(4,3,2,1,6,5,4,3));
BB = Tensor{5:8, 9:12}(rand(6,5,4,3,5,4,3,2));

@info "Testing (re)construction"
A_ = A(3,2);
@test (lspaces(A_), rspaces(A_)) == ((3,2), (3,2))
@btime A_ = ($A)(3,2);      # 1.0μs (14 allocations)
@btime A_ = ($A)((3,2));      # 1.0μs (14 allocations)
@btime A_ = ($A)((3,2),(3,2));      # 1.0μs (14 allocations)

A_ = A((5,6),(8,7));
@test (lspaces(A_), rspaces(A_)) == ((5,6), (8,7))
@btime A_ = ($A)((5,6),(8,7));  # 1.0μs (14 allocations)


@info "testing getindex"
T = Tensor{(5,2,3),(10,60)}(randn(2,3,4,5,6));
S = T[:,2,:,4,:];
@test (lspaces(S), rspaces(S)) == ((5,3), (60,))
@btime ($T)[:,2,:,4,:];             # 89 ns, 140 ns (1 alloc)
@btime AlgebraicTensors.getindex_($T, (:,2,:,4,:)); # 140 ns (1 alloc)


@info "testing transpose"
T = Tensor(randn(2,3,4,5,6), (2,5,3), (1,5));
S = transpose(T, 5);
@test (lspaces(S), rspaces(S)) == ((2,3,5), (1,5))
@test size(S) == (2,4,6,5,3)
@btime transpose($T, 5);		# 960 ns

S = transpose(T, (5,2));
@test (lspaces(S), rspaces(S)) == ((3,5), (1,2,5))
@test size(S) == (4,6,5,2,3)
@btime transpose($T, (5,2));	# 890 ns


@info "testing multiplication"
@test A*B == C
@test_throws DimensionMismatch A*B(2,1)
@test A(9,5)*B(9,5) == C(9,5)

CC = AA*BB;
@test (lspaces(CC), rspaces(CC)) == ((1,2,3,4), (9,10,11,12))

@info "Benchmarking small multiplication."  #  Expect 174 ns (7 allocations: 416 bytes)
@btime $A*$B;

@info "Benchmarking large multiplication."    #  Expect 105 μs (2 allocations: 23 KiB)
@btime $AA*$BB;


@info "Benchmarking small trace."  # Expect 16 ns (0 allocations)
@btime tr($R);

@info "Testing small partial trace."	# Expect 160 ns (4 allocations)
@btime tr($R, 2);


@info "Combination tests"
@test size(A + B') == (1,2,3,4)
@test R * inv(R) ≈ Tensor(reshape(I(12), (3,4,3,4)), 1:2)


# tr(Matrix(R)) takes only 14 ns.

