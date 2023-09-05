using AlgebraicTensors
using BenchmarkTools
using LinearAlgebra: I, tr
using Test


# Times shown are for my HP ZBook Firefly G10, i7-1360P

A = Tensor{(1,2)}(reshape(1.0:24, (1,2,3,4)));
B = Tensor{1:2}(reshape(1.0:24, (3,4,1,2)));
C = Tensor{(1,2)}(reshape([1222.0, 1300.0, 2950.0, 3172.0], (1,2,1,2)));
R = Tensor{(1,2)}(rand(3,4,3,4));
R_ = Tensor{(1,2),(2,1)}(permutedims(R.data, (1,2,4,3)));

AA = Tensor{1:4, 5:8}(rand(4,3,2,1,6,5,4,3));
BB = Tensor{5:8, 9:12}(rand(6,5,4,3,5,4,3,2));
RR = Tensor{(1,2)}(rand(10,20,10,20));
RR_ = Tensor{(1,2),(2,1)}(rand(10,20,20,10));

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
@btime ($T)[:,2,:,4,:];             # 89 ns (1 alloc)


@info "testing =="
A_ = A((2,1));
A__ = Tensor{(2,1)}(permutedims(A.data, (2,1,4,3)));
AA_ = AA((4,3,2,1),(8,7,6,5));
AA__ = Tensor{(4,3,2,1),(8,7,6,5)}(permutedims(AA.data, (4,3,2,1,8,7,6,5)));

@test A == A
@test A != A_
@test A == A__
@btime $A == $A;     # 19 ns
@btime $A == $A__;     # 33 ns (with @inbounds)

@test AA == AA
@test AA != AA_
@test AA == AA__
@btime $AA == $AA;      # 6.3 μs
@btime $AA == $AA__;    # 8.8 μs (with @inbounds)



@info "testing +,-"
X = Tensor{(3,5),(4,8)}(reshape(1.0:1.0:24, (1,2,3,4)));
Y = permutedims(2*X, (2,1,4,3))((5,3),(8,4));
@test X+X == Y;
@test X+Y == 3*X;
@btime $X+$X;   # 139 ns
@btime $X+$Y;   # 81 ns

XX = Tensor{(4,3,2,1), (8,7,6,5)}(rand(1,2,3,4,3,4,5,6));
@btime $AA+$AA;     #27 μs
@btime $AA+$XX;     #15 μs



@info "testing transpose"
T = Tensor{(2,5,3),(1,5)}(randn(2,3,4,5,6));
S = transpose(T, 5);
@test (lspaces(S), rspaces(S)) == ((2,3,5), (1,5))
@test size(S) == (2,4,6,5,3)
@btime transpose($T, 5);		# 675 ns

S = transpose(T, (5,2));
@test (lspaces(S), rspaces(S)) == ((3,5), (1,2,5))
@test size(S) == (4,6,5,2,3)
@btime transpose($T, (5,2));	# 670 ns


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


@info "Benchmarking trace."
@btime tr($R);          # 16 ns (0 allocations)
@btime tr($R_);         # 25 ns (0 allocations)
@btime tr($RR);         # 86 ns
@btime tr($RR_);         # 129 ns

@info "Testing partial trace."	
@btime tr($R, 2);            # 160 ns (4 allocations)
@btime tr($R_, 2);            # 176 ns (4 allocations)
@btime tr($R, Val((2,)));    # 51 ns (1 allocations)

@btime tr($RR, 2);           # 1.2 μs (4 allocations)
@btime tr($RR, Val((2,)));   # 1.0 μs (1 allocations)

@info "Benchmarking marginal." 
@btime marginal($R, 1);          # Expect 150 ns (0 allocations)


@info "Testing eig"
e = eigvals(R);
e_ = eigvals(R_)
@test e == e_;

V = eigvecs(R);
V_ = eigvecs(R_);
@test Matrix(V) == Matrix(V_)
@test Matrix(V) == eigvecs(Matrix(R))


@info "Testing svd"
s = svdvals(R)
s_ = svdvals(R_)
@test s == s_ == svdvals(Matrix(R))

(U,S,Vt) = svd(R);
result = svd(Matrix(R));
@test Matrix(U) == result.U
@test Matrix(Vt) == result.Vt
@test S == result.S

(U_,S_,Vt_) = svd(R_);
@test U ≈ U_
@test S ≈ S_
@test Vt ≈ Vt_


@info "Combination tests"
@test size(A + B') == (1,2,3,4)
@test R * inv(R) ≈ Tensor{1:2}(reshape(I(12), (3,4,3,4)))

# tr(Matrix(R)) takes only 14 ns.

