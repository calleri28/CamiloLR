import math

import lac.vector as vector_ops
import lac.matrix as matrix_ops
from lac import Matrix, Vector

VEC1 = Vector([1, 2, -3])
VEC2 = Vector([0, 0, 0])
VEC3 = Vector([math.pi, 1 / math.pi, math.e])

ALL_VECTORS = [VEC1, VEC2, VEC3]

ALL_SCALARS = [0, 1, -1, -math.pi, 0.5]

MAT1 = Matrix([Vector([1, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1])])

REDUCED_MATRICES = [
    (
        Matrix(
            [
                Vector([2, 5, 8, 7]),
                Vector([5, 2, 2, 8]),
                Vector([7, 5, 6, 6]),
                Vector([5, 4, 4, 8]),
            ]
        ),
        Matrix(
            [
                Vector([7, 5, 6, 6]),
                Vector([0, 3.5714285714285716, 6.285714285714286, 5.285714285714286]),
                Vector([0, 0, -1.04, 3.08]),
                Vector([0, 0, 0, 7.46153846153846]),
            ]
        ),
    ),
    (
        Matrix([Vector([1, 2]), Vector([3, 9])]),
        Matrix([Vector([3, 9]), Vector([0, -1])]),
    ),
    (
        Matrix([Vector([2, 1, 1]), Vector([4, -6, 0]), Vector([-2, 7, 2])]),
        Matrix([Vector([4, -6, 0]), Vector([0, 4, 1]), Vector([0, 0, 1])]),
    ),
]

ALL_MATRICES = [MAT1] + [orig for orig, _ in REDUCED_MATRICES]


def assert_vectors_almost_equal(v1, v2):
    assert vector_ops.almost_equal(v1, v2)


def assert_matrices_almost_equal(m1, m2):
    assert matrix_ops.almost_equal(m1, m2)
