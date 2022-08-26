import copy
import itertools
import math
import unittest

import lac.vector as vector_ops
from lac import Vector
from lac import PRECISION

from lac import testing_utils as utils


class TestVector(unittest.TestCase):
    def make_random(self):
        dim = 3
        vec = Vector.make_random(dim)
        self.assertEqual(vec.norm, 1, PRECISION)

    def test_make_zero(self):
        dim = 3
        vec = Vector.make_zero(dim)
        self.assertEqual(vec.norm, 0, PRECISION)
        self.assertEqual(vec.dim, dim)

    def test_make_unitary(self):
        for vec in utils.ALL_VECTORS:
            if vec.norm > 0:
                v = Vector.make_unitary(vec)
                self.assertAlmostEqual(v.norm, 1, PRECISION)
                self.assertEqual(v.dim, vec.dim)
                utils.assert_vectors_almost_equal(vec.norm * v, vec)

    def test_getitem_int(self):
        for vec in utils.ALL_VECTORS:
            for i in range(vec.dim):
                self.assertIsInstance(vec[i], (int, float))

    def test_getitem(self):
        for vec in utils.ALL_VECTORS:
            for start in range(0, vec.dim + 1):
                for stop in range(start + 1, vec.dim + 2):
                    for step in range(1, stop):
                        v = vec[start:step:stop]
                        self.assertIsInstance(v, Vector)

    def test_setitem_int(self):
        all_vecs = copy.deepcopy(utils.ALL_VECTORS)
        for vec, k in itertools.product(all_vecs, utils.ALL_SCALARS):
            for i in range(vec.dim):
                vec[i] = k
                self.assertIsInstance(vec[i], (int, float))
                self.assertAlmostEqual(vec[i], k)

    def test_setitem_slice_to_int(self):
        all_vecs = copy.deepcopy(utils.ALL_VECTORS)
        for vec, k in itertools.product(all_vecs, utils.ALL_SCALARS):
            for start in range(0, vec.dim + 1):
                for stop in range(start + 1, vec.dim + 2):
                    for step in range(1, stop):
                        vec[start:stop:step] = k
                        length = math.ceil((min(stop, len(vec)) - start) / step)
                        real = vec[start:stop:step]
                        expected = Vector([k] * length)
                        self.assertIsInstance(real, Vector)
                        self.assertEqual(real, expected)

    def test_setitem_slice_to_sequence(self):
        all_vecs = copy.deepcopy(utils.ALL_VECTORS)
        for vec, k in itertools.product(all_vecs, utils.ALL_SCALARS):
            for start in range(0, vec.dim + 1):
                for stop in range(start + 1, vec.dim + 2):
                    for step in range(1, stop):
                        length = math.ceil((stop - start) / step)
                        expected = Vector([k] * length)
                        vec[start:stop:step] = expected
                        real = vec[start:stop:step]
                        self.assertIsInstance(real, Vector)
                        self.assertEqual(real, expected)

    def test_dim(self):
        components = [3, 2, 1]
        vec = Vector(components)
        self.assertEqual(len(components), vec.dim)

    def test_add(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            utils.assert_vectors_almost_equal(v1 + v2, vector_ops.add(v1, v2))

    def test_scalar_multiply(self):
        for v, k in itertools.product(utils.ALL_VECTORS, utils.ALL_SCALARS):
            utils.assert_vectors_almost_equal(k * v, vector_ops.scale(v, k))

    def test_dot(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            self.assertAlmostEqual(v1 @ v2, vector_ops.dot(v1, v2), PRECISION)


class TestVectorOps(unittest.TestCase):
    def test_build_unit_vector(self):
        for vec in utils.ALL_VECTORS:
            if vec.norm > 0:
                unit = vector_ops.build_unit_vector(vec)
                self.assertAlmostEqual(unit.norm, 1, PRECISION)
                utils.assert_vectors_almost_equal(vec.norm * unit, vec)

    def test_project(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            if v1.norm > 0 and v2.norm > 0:
                pass
                # vec = vector_ops.project(v1, v2)
                # value = v1.norm * math.cos(vector_ops.angle_between(v1, v2))
                # self.assertIsInstance(vec, Vector)
                # self.assertAlmostEqual(vec.norm, abs(value))

    def test_scale(self):
        for v, k in itertools.product(utils.ALL_VECTORS, utils.ALL_SCALARS):
            if v.norm != 0 and k > 0:
                unit = vector_ops.build_unit_vector(v)
                utils.assert_vectors_almost_equal(vector_ops.scale(unit, v.norm), v)
                utils.assert_vectors_almost_equal(
                    vector_ops.build_unit_vector(vector_ops.scale(v, k)), unit
                )
                utils.assert_vectors_almost_equal(
                    vector_ops.scale(v, (1 / v.norm)), unit
                )

    def test_add_commutative(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            vec1 = vector_ops.add(v1, v2)
            vec2 = vector_ops.add(v2, v1)
            utils.assert_vectors_almost_equal(vec1, vec2)

    def test_add_associative(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            for v3 in utils.ALL_VECTORS:
                vec1 = vector_ops.add(vector_ops.add(v1, v2), v3)
                vec2 = vector_ops.add(v1, vector_ops.add(v2, v3))
                utils.assert_vectors_almost_equal(vec1, vec2)

    def test_subtract(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            utils.assert_vectors_almost_equal(
                vector_ops.subtract(v1, v2), vector_ops.add(v1, -v2)
            )

    def test_dot_commutative(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            self.assertAlmostEqual(vector_ops.dot(v1, v2), vector_ops.dot(v2, v1))

    def test_dot_distributive(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            for v3 in utils.ALL_VECTORS:
                f1 = vector_ops.dot(v1, v2 + v3)
                f2 = vector_ops.dot(v1, v2) + vector_ops.dot(v1, v3)
                self.assertAlmostEqual(f1, f2, PRECISION)

    def test_dot_bilinear(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            for v3 in utils.ALL_VECTORS:
                for k in utils.ALL_SCALARS:
                    f1 = vector_ops.dot(v1, (k * v2) + v3)
                    f2 = k * (vector_ops.dot(v1, v2)) + vector_ops.dot(v1, v3)
                    self.assertAlmostEqual(f1, f2, PRECISION)

    def test_dot_scalar_multiplitcation(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            for k1, k2 in itertools.product(utils.ALL_SCALARS, utils.ALL_SCALARS):
                f1 = vector_ops.dot(k1 * v1, k2 * v2)
                f2 = k1 * k2 * vector_ops.dot(v1, v2)
                self.assertAlmostEqual(f1, f2, PRECISION)

    def test_dot_not_associative(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            for v3 in utils.ALL_VECTORS:
                with self.assertRaises(AttributeError):
                    vector_ops.dot(vector_ops.dot(v1, v2), v3)

    def test_dot_orthogonal(self):
        v1 = Vector([0, 1, 0])
        v2 = Vector([1, 0, 0])
        self.assertAlmostEqual(vector_ops.dot(v1, v2), 0, PRECISION)

    def test_angle_between_parallel(self):
        v1 = Vector([1, 0, 0])
        v2 = Vector([2, 0, 0])
        self.assertAlmostEqual(vector_ops.angle_between(v1, v2), 0, PRECISION)

    def test_angle_between_antiparallel(self):
        v1 = Vector([-1, 0, 0])
        v2 = Vector([2, 0, 0])
        self.assertAlmostEqual(vector_ops.angle_between(v1, v2), math.pi, PRECISION)

    def test_angle_between_orthogonal(self):
        v1 = Vector([0, 1, 0])
        v2 = Vector([1, 0, 0])
        self.assertAlmostEqual(vector_ops.angle_between(v1, v2), math.pi / 2, PRECISION)

    def test_dot_angle_equality(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            if v1.norm > 0 and v2.norm > 0:
                f1 = vector_ops.dot(v1, v2)
                f2 = v1.norm * v2.norm * math.cos(vector_ops.angle_between(v1, v2))
                self.assertAlmostEqual(f1, f2, PRECISION)

    def test_cross_parallel(self):
        v1 = Vector([1, 0, 0])
        v2 = Vector([2, 0, 0])
        utils.assert_vectors_almost_equal(vector_ops.cross(v1, v2), Vector.make_zero(3))

    def test_cross_orthogonal(self):
        v1 = Vector([0, 1, 0])
        v2 = Vector([1, 0, 0])
        utils.assert_vectors_almost_equal(vector_ops.cross(v1, v2), Vector([0, 0, -1]))

    def test_cross_angle_equality(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            if v1.norm > 0 and v2.norm > 0:
                theta = vector_ops.angle_between(v1, v2)
                f1 = vector_ops.cross(v1, v2).norm
                f2 = v1.norm * v2.norm * abs(math.sin(theta))
                self.assertAlmostEqual(f1, f2, PRECISION)

    def test_cross_self(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            if v1 == v2:
                vec = vector_ops.cross(v1, v2)
                utils.assert_vectors_almost_equal(vec, Vector.make_zero(3))

    def test_cross_anticommutative(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            vec1 = vector_ops.cross(v1, v2)
            vec2 = vector_ops.cross(v2, v1)
            utils.assert_vectors_almost_equal(vec1, -vec2)

    def test_cross_distributive(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            for v3 in utils.ALL_VECTORS:
                vec1 = vector_ops.cross(v1, v2 + v3)
                vec2 = vector_ops.cross(v1, v2) + vector_ops.cross(v1, v3)
                utils.assert_vectors_almost_equal(vec1, vec2)

    def test_cross_scalar_multiply(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            for k in utils.ALL_SCALARS:
                vec1 = vector_ops.cross(k * v1, v2)
                vec2 = vector_ops.cross(v1, k * v2)
                vec3 = k * vector_ops.cross(v1, v2)
                for a, b in itertools.combinations([vec1, vec2, vec3], 2):
                    utils.assert_vectors_almost_equal(a, b)

    def test_cross_jacobi_identity(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            for v3 in utils.ALL_VECTORS:
                vec1 = vector_ops.cross(v1, vector_ops.cross(v2, v3))
                vec2 = vector_ops.cross(v2, vector_ops.cross(v3, v1))
                vec3 = vector_ops.cross(v3, vector_ops.cross(v1, v2))
                utils.assert_vectors_almost_equal(
                    vec1 + vec2 + vec3, Vector.make_zero(3)
                )

    def test_cross_parallelepiped_volume(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            for v3 in utils.ALL_VECTORS:
                f1 = v1 @ vector_ops.cross(v2, v3)
                f2 = v2 @ vector_ops.cross(v3, v1)
                f3 = v3 @ vector_ops.cross(v1, v2)
                for a, b in itertools.combinations([f1, f2, f3], 2):
                    self.assertAlmostEqual(a, b, PRECISION)

    def test_cross_dot_relationship(self):
        for v1, v2 in itertools.product(utils.ALL_VECTORS, utils.ALL_VECTORS):
            for v3 in utils.ALL_VECTORS:
                vec1 = vector_ops.cross(v1, vector_ops.cross(v2, v3))
                vec2 = ((v1 @ v3) * v2) - ((v1 @ v2) * v3)
                utils.assert_vectors_almost_equal(vec1, vec2)
