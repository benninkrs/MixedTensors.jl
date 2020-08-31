using MultiMatrices
using BenchmarkTools
using LinearAlgebra: I, tr
using Test

A = MultiMatrix(reshape(1.0:24, (1,2,3,4)))
B = MultiMatrix(reshape(1.0:24, (3,4,1,2)))
C = MultiMatrix(reshape([1222.0, 1300.0, 2950.0, 3172.0], (1,2,1,2)))
R = MultiMatrix(rand(3,4,3,4))

AA = MultiMatrix(rand(4,3,2,1,6,5,4,3))
BB = MultiMatrix(rand(6,5,4,3,5,4,3,2))

@test A*B == C
@test_throws DimensionMismatch A*B(2,1)
@test A(9,5)*B(9,5) == C(9,5)
@test size(A + B') == (1,2,3,4)
@test R * inv(R) ≈ MultiMatrix(reshape(I(12), (3,4,3,4)))

@info "Testing construction.  Expect 520 ns (7 allocations: 704 bytes)"
@btime ($A)(2,3);

@info "Testing small multiplication.  Expect 700 ns (7 allocations: 704 bytes)"
@btime $A*$B;

@info "Testing large multiplication.  Expect 49 μs (15 allocations: 23 KiB)"
@btime $AA*$BB;

# tr(Matrix(R)) takes only 12 ns.
@info "Testing small trace.  Expect 60 ns (3 allocations: 112 bytes)"
@btime tr($R);

@info "Testing small partial trace.  Expect ? ns (? allocations: ? bytes)"
@btime tr($R, 2);
