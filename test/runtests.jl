using MultiMatrices

A = MultiMatrix(reshape(1:24, (1,2,3,4)))
B = MultiMatrix(reshape(1:24, (3,4,1,2)))
C = MultiMatrix(reshape([1222, 1300, 2950, 3172], (1,2,1,2)))
S = MultiMatrix(rand(2,3,2,3))

@test A*B == C
@test_throws DimensionMismatch A*B(2,1)
@test A(9,5)*B(9,5) == C(9,5)
@test size(A + B') = (1,2,3,4)
@test S * inv(S) ≈ MultiMatrix(reshape(I(6), (2,3,2,3)))
