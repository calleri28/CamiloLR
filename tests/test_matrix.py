import copy
import itertools
import math
import unittest

import lac.matrix as matrix_ops
import lac.vector as vector_ops
from lac import Matrix, Vector, PRECISION

import lac.testing_utils as utils


class TestMatrix(unittest.TestCase):
    def test_from_columnvectors(self):
        mat = Matrix.from_columnvectors(
            [Vector([1, 0]), Vector([0, 1]), Vector([0, 0])]
        )
        self.assertTupleEqual(mat.shape, (2, 3))

    def test_make_random(self):
        for n, m in itertools.combinations(range(2, 10), 2):
            mat = Matrix.make_random(n, m)
            self.assertTupleEqual(mat.shape, (n, m))

    def test_make_identity(self):
        for n, m in itertools.combinations(range(2, 10), 2):
            mat = Matrix.make_random(n, m)
            eye_right = Matrix.make_identity(m, m)
            eye_left = Matrix.make_identity(n, n)
            utils.assert_matrices_almost_equal(mat @ eye_right, mat)
            utils.assert_matrices_almost_equal(eye_left @ mat, mat)

    def test_make_zero(self):
        for n, m in itertools.combinations(range(2, 10), 2):
            mat = Matrix.make_random(n, m)
            zero_right = Matrix.make_zero(m, m)
            zero_left = Matrix.make_zero(n, m)
            utils.assert_matrices_almost_equal(mat @ zero_right, zero_left)
            utils.assert_matrices_almost_equal(zero_left.T @ mat, zero_right)

    def test_iteration_by_rows(self):
        for mat in utils.ALL_MATRICES:
            for i, (row1, row2) in enumerate(zip(mat, mat.iterrows())):
                self.assertEqual(row1.dim, mat.num_columns)
                self.assertEqual(row2.dim, mat.num_columns)
                self.assertEqual(row1, row2)
            self.assertEqual(i + 1, mat.num_rows)

    def test_iteration_by_columns(self):
        for mat in utils.ALL_MATRICES:
            for i, col in enumerate(mat.itercolumns()):
                self.assertEqual(col.dim, mat.num_rows)
            self.assertEqual(i + 1, mat.num_columns)

    def test_getitem_single_value(self):
        for mat in utils.ALL_MATRICES:
            for i, j in itertools.product(range(mat.num_rows), range(mat.num_columns)):
                self.assertIsInstance(mat[i, j], (int, float))

    def test_getitem_row(self):
        for mat in utils.ALL_MATRICES:
            for i, row in enumerate(mat.iterrows()):
                self.assertIsInstance(mat[i], Vector)
                vector_ops.almost_equal(mat[i], row)

    def test_getitiem_slice_rows(self):
        for mat in utils.ALL_MATRICES:
            for start in range(0, mat.num_rows):
                for stop in range(start + 1, mat.num_rows + 1):
                    for step in range(1, stop):
                        m1 = mat[start:stop:step]
                        m2 = mat[start:stop:step, :]
                        length = math.ceil((stop - start) / step)
                        self.assertIsInstance(m1, Matrix)
                        self.assertIsInstance(m2, Matrix)
                        self.assertEqual(m1, m2)
                        self.assertEqual(m1.shape, (length, mat.num_columns))

    def test_getitem_column(self):
        for mat in utils.ALL_MATRICES:
            for i, col in enumerate(mat.itercolumns()):
                self.assertIsInstance(mat[:, i], Vector)
                vector_ops.almost_equal(mat[:, i], col)

    def test_getitiem_slice_columns(self):
        for mat in utils.ALL_MATRICES:
            for start in range(0, mat.num_columns):
                for stop in range(start + 1, mat.num_columns + 1):
                    for step in range(1, stop):
                        m = mat[:, start:stop:step]
                        length = math.ceil((stop - start) / step)
                        self.assertIsInstance(m, Matrix)
                        self.assertEqual(m.shape, (mat.num_rows, length))

    def test_getitem(self):
        for mat in utils.ALL_MATRICES:
            STOP = min(mat.num_rows, mat.num_columns)
            for start in range(0, STOP):
                for stop in range(start + 1, STOP + 1):
                    for step in range(1, stop):
                        m = mat[start:stop:step, start:stop:step]
                        self.assertIsInstance(m, Matrix)
                        if m.shape == mat.shape:
                            utils.assert_matrices_almost_equal(m, mat)

    def test_setitem_row_as_int(self):
        all_matrices = copy.deepcopy(utils.ALL_MATRICES)
        for mat, k in itertools.product(all_matrices, utils.ALL_SCALARS):
            vec = Vector([k] * mat.num_columns)
            for start in range(0, mat.num_rows):
                mat[start] = k
                utils.assert_vectors_almost_equal(mat[start], vec)

    def test_setitem_row_as_sequence(self):
        factories = [lambda x: x, lambda x: x.components, list, tuple]
        all_matrices = copy.deepcopy(utils.ALL_MATRICES)
        for mat, k in itertools.product(all_matrices, utils.ALL_SCALARS):
            vec = Vector([k] * mat.num_columns)
            for start in range(0, mat.num_rows):
                for fac in factories:
                    mat[start] = fac(vec)
                    utils.assert_vectors_almost_equal(mat[start], vec)

    def test_setitem_row_slice_as_int(self):
        all_matrices = copy.deepcopy(utils.ALL_MATRICES)
        for mat, k in itertools.product(all_matrices, utils.ALL_SCALARS):
            vec = Vector([k] * mat.num_columns)
            for start in range(0, mat.num_rows):
                for stop in range(start + 1, mat.num_rows + 1):
                    for step in range(1, stop):
                        mat[start:stop:step] = k
                        for i in range(start, stop, step):
                            utils.assert_vectors_almost_equal(mat[i], vec)

    def test_setitem_row_slice_as_sequence(self):
        factories = [lambda x: x, lambda x: x.components, list, tuple]
        all_matrices = copy.deepcopy(utils.ALL_MATRICES)
        for mat, k in itertools.product(all_matrices, utils.ALL_SCALARS):
            vec = Vector([k] * mat.num_columns)
            for start in range(0, mat.num_rows):
                for stop in range(start + 1, mat.num_rows + 1):
                    for step in range(1, stop):
                        for fac in factories:
                            mat[start:stop:step] = fac(vec)
                            for i in range(start, stop, step):
                                utils.assert_vectors_almost_equal(mat[i], vec)

    def test_setitem_column_as_int(self):
        all_matrices = copy.deepcopy(utils.ALL_MATRICES)
        for mat, k in itertools.product(all_matrices, utils.ALL_SCALARS):
            vec = Vector([k] * mat.num_rows)
            for start in range(0, mat.num_columns):
                mat[:, start] = k
                utils.assert_vectors_almost_equal(mat[:, start], vec)

    def test_setitem_column_as_sequence(self):
        factories = [lambda x: x, lambda x: x.components, list, tuple]
        all_matrices = copy.deepcopy(utils.ALL_MATRICES)
        for mat, k in itertools.product(all_matrices, utils.ALL_SCALARS):
            vec = Vector([k] * mat.num_rows)
            for start in range(0, mat.num_columns):
                for fac in factories:
                    mat[:, start] = fac(vec)
                    utils.assert_vectors_almost_equal(mat[:, start], vec)

    def test_setitem_column_slice_as_int(self):
        all_matrices = copy.deepcopy(utils.ALL_MATRICES)
        for mat, k in itertools.product(all_matrices, utils.ALL_SCALARS):
            vec = Vector([k] * mat.num_rows)
            for start in range(0, mat.num_columns):
                for stop in range(start + 1, mat.num_columns + 1):
                    for step in range(1, stop):
                        mat[:, start:stop:step] = k
                        for i in range(start, stop, step):
                            utils.assert_vectors_almost_equal(mat[:, i], vec)

    def test_setitem_column_slice_as_sequence(self):
        factories = [lambda x: x, lambda x: x.components, list, tuple]
        all_matrices = copy.deepcopy(utils.ALL_MATRICES)
        for mat, k in itertools.product(all_matrices, utils.ALL_SCALARS):
            vec = Vector([k] * mat.num_columns)
            for start in range(0, mat.num_rows):
                for stop in range(start + 1, mat.num_rows + 1):
                    for step in range(1, stop):
                        for fac in factories:
                            mat[:, start:stop:step] = fac(vec)
                            for i in range(start, stop, step):
                                utils.assert_vectors_almost_equal(mat[:, i], vec)

    def test_setitem_as_int(self):
        all_matrices = copy.deepcopy(utils.ALL_MATRICES)
        for mat, k in itertools.product(all_matrices, utils.ALL_SCALARS):
            for i in range(mat.num_rows):
                for j in range(mat.num_columns):
                    mat[i, j] = k
                    self.assertEqual(mat[i, j], k)

    def test_setitem_as_matrix(self):
        pass

    def test_transponse_involution(self):
        for mat in utils.ALL_MATRICES:
            utils.assert_matrices_almost_equal(mat.T.T, mat)

    def test_transpose_linearity(self):
        pass

    def test_transpose_multiplication(self):
        for m1, m2 in itertools.product(utils.ALL_MATRICES, utils.ALL_MATRICES):
            if m1.num_columns == m2.num_rows:
                utils.assert_matrices_almost_equal((m1 @ m2).T, m2.T @ m1.T)

    def test_transpose_cyclic(self):
        pass

    def test_transpose_det(self):
        pass

    def test_transpose_inverse(self):
        pass

    def test_transponse_eigenvalues(self):
        pass

    def test_norm_scale(self):
        pass

    def test_norm_addition(self):
        pass

    def test_norm_possitive(self):
        pass

    def test_norm_zero(self):
        pass

    def test_determinant_execution(self):
        for mat in utils.ALL_MATRICES:
            if mat.num_rows == mat.num_columns:
                _ = mat.determinant
            else:
                with self.assertRaises(RuntimeError):
                    _ = mat.determinant
    
    def test_determinant_product(self):
        for mat1, mat2 in itertools.product(utils.ALL_MATRICES, utils.ALL_MATRICES):
            if mat1.shape == mat2.shape and mat1.num_rows == mat1.num_columns:
                prod = mat1 @ mat2
                self.assertAlmostEqual(prod.determinant, mat1.determinant * mat2.determinant, PRECISION)
                self.assertAlmostEqual(prod.determinant, (mat2 @ mat1).determinant)

    def test_determinant_transpose(self):
        for mat in utils.ALL_MATRICES:
            if mat.num_rows == mat.num_columns:
                self.assertAlmostEqual(mat.T.determinant, mat.determinant)

    def test_determinant_scalar_multiply(self):
        for k, mat in itertools.product(utils.ALL_SCALARS, utils.ALL_MATRICES):
            if mat.num_rows == mat.num_columns:
                self.assertAlmostEqual((k * mat).determinant, k**mat.num_rows * mat.determinant)


    def test_inverse(self):
        pass

    def test_trace(self):
        pass

    def test_trace_identity(self):
        pass

    def test_matmul(self):
        for mat1, mat2 in itertools.product(utils.ALL_MATRICES, utils.ALL_MATRICES):
            if mat1.num_columns == mat2.num_rows:
                utils.assert_matrices_almost_equal(
                    mat1 @ mat2, matrix_ops.matrix_multiply(mat1, mat2)
                )

    def test_add(self):
        for mat1, mat2 in itertools.product(utils.ALL_MATRICES, utils.ALL_MATRICES):
            if mat1.shape == mat2.shape:
                utils.assert_matrices_almost_equal(
                    mat1 + mat2, matrix_ops.add(mat1, mat2)
                )

    def test_scalar_multiply(self):
        for k, mat in itertools.product(utils.ALL_SCALARS, utils.ALL_MATRICES):
            utils.assert_matrices_almost_equal(k * mat, matrix_ops.scale(mat, k))

    def test_negation(self):
        for mat in utils.ALL_MATRICES:
            utils.assert_matrices_almost_equal(-mat, matrix_ops.scale(mat, -1))

    def test_subtraction(self):
        for mat1, mat2 in itertools.product(utils.ALL_MATRICES, utils.ALL_MATRICES):
            if mat1.shape == mat2.shape:
                utils.assert_matrices_almost_equal(
                    mat1 - mat2, matrix_ops.subtract(mat1, mat2)
                )


class TestMatrixOps(unittest.TestCase):
    def test_scale(self):
        for m, k in itertools.product(utils.ALL_MATRICES, utils.ALL_SCALARS):
            mat = matrix_ops.scale(m, k)
            self.assertTupleEqual(m.shape, mat.shape)
            for i in range(mat.num_rows):
                for j in range(mat.num_columns):
                    self.assertAlmostEqual(k * m[i, j], mat[i, j])

    def test_add(self):
        for m1, m2 in itertools.product(utils.ALL_MATRICES, utils.ALL_MATRICES):
            if m1.shape == m2.shape:
                mat = matrix_ops.add(m1, m2)
                for i in range(mat.num_rows):
                    for j in range(mat.num_columns):
                        self.assertAlmostEqual(m1[i, j] + m2[i, j], mat[i, j])

    def test_subtract(self):
        for m1, m2 in itertools.product(utils.ALL_MATRICES, utils.ALL_MATRICES):
            if m1.shape == m2.shape:
                mat = matrix_ops.subtract(m1, m2)
                for i in range(mat.num_rows):
                    for j in range(mat.num_columns):
                        self.assertAlmostEqual(m1[i, j] - m2[i, j], mat[i, j])

    def test_vector_multiply_from_right(self):
        for mat in utils.ALL_MATRICES:
            for vec in utils.ALL_VECTORS:
                if mat.num_columns == vec.dim:
                    for b in (True, False):
                        v = matrix_ops.vector_multiply(mat, vec, from_left=b)
                        self.assertIsInstance(v, Vector)
                        if b:
                            self.assertEqual(v.dim, mat.num_columns)
                        else:
                            self.assertEqual(v.dim, mat.num_rows)
                else:
                    with self.assertRaises(ValueError):
                        _ = matrix_ops.vector_multiply(mat, v)

    def test_matrix_multiply(self):
        for m1, m2 in itertools.product(utils.ALL_MATRICES, utils.ALL_MATRICES):
            if m1.num_columns == m2.num_rows:
                m = matrix_ops.matrix_multiply(m1, m2)
                self.assertIsInstance(m, Matrix)
            else:
                with self.assertRaises(ValueError):
                    _ = matrix_ops.matrix_multiply(m1, m2)

    def test_gaussian_elimination(self):
        pass

