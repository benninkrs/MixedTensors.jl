using MixedTensors
using BenchmarkTools
using LinearAlgebra: I, tr
using Test




A = Tensor(reshape(1.0:24, (1,2,3,4)), (1,2))
B = Tensor(reshape(1.0:24, (3,4,1,2)), 1:2)
C = Tensor(reshape([1222.0, 1300.0, 2950.0, 3172.0], (1,2,1,2)), 1:2)
R = Tensor(rand(3,4,3,4), 1:2);

AA = Tensor(rand(4,3,2,1,6,5,4,3), 1:4);
BB = Tensor(rand(6,5,4,3,5,4,3,2), 1:4);

@info "Testing construction.  Expect 520 ns (7 allocations: 704 bytes)"
@btime ($A)(2,3);


@info "testing getindex"
T = Tensor(randn(2,3,4,5,6), (5,2,3), (10,60));
S = T[:,2,:,4,:];
@btime ($T)[:,2,:,4,:];


@info "testing transpose"
T = Tensor(randn(2,3,4,5,6), (2,5,3), (1,5));
S = transpose(T, 5);
@test spaces(S) == ((2,3,5), (1,5))
@test size(S) == (2,4,6,5,3)

S = transpose(T, (5,2));
@test spaces(S) == ((3,5), (1,2,5))
@test size(S) == (4,6,5,2,3)


@test A*B == C
@test_throws DimensionMismatch A*B(2,1)
@test A(9,5)*B(9,5) == C(9,5)
@test size(A + B') == (1,2,3,4)
@test R * inv(R) ≈ Tensor(reshape(I(12), (3,4,3,4)), 1:2)

@info "Testing small multiplication.  Expect 700 ns (7 allocations: 704 bytes)"
@btime $A*$B;

@info "Testing large multiplication.  Expect 49 μs (15 allocations: 23 KiB)"
@btime $AA*$BB;  #SLOW!

# tr(Matrix(R)) takes only 14 ns.
@info "Testing small trace.  Expect 19 ns (2 allocations: 96 bytes)"
@btime tr($R);

@info "Testing small partial trace.  Expect 473 ns (12 allocations: 816 bytes)"
@btime tr($R, 2);		# SLOW!
