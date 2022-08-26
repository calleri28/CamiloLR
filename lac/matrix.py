from __future__ import annotations

import copy
import typing as t
from array import array
from collections.abc import Iterable

import lac.vector as vector_ops
from lac import Vector, PRECISION


class Matrix:
    @classmethod
    def from_columnvectors(cls, vectors: t.Iterable[Vector]):
        vectors = list(vectors)
        _validate_vector_dimensions(vectors)
        rowvectors = [Vector(v[i] for v in vectors) for i in range(vectors[0].dim)]
        return cls(rowvectors)

    @classmethod
    def make_random(cls, num_rows: int, num_columns: int):
        """A Matrix built out of random unit row vectors. """
        return cls(Vector.make_random(num_columns) for _ in range(num_rows))

    @classmethod
    def make_identity(cls, num_rows: int, num_columns: int):
        rowvectors = _make_identity_rowvectors(num_rows, num_columns)
        return cls(rowvectors)

    @classmethod
    def make_zero(cls, num_rows: int, num_columns: int):
        return cls(Vector.make_zero(num_columns) for _ in range(num_rows))

    @classmethod
    def make_row_switching(cls, num_rows: int, num_columns: int, i: int, j: int):
        rowvectors = _make_identity_rowvectors(num_rows, num_columns)
        rowvectors[i], rowvectors[j] = rowvectors[j], rowvectors[i]
        return cls(rowvectors)

    @classmethod
    def make_row_multiplying(
        cls, num_rows: int, num_columns: int, i: int, m: t.Union[int, float]
    ):
        rowvectors = _make_identity_rowvectors(num_rows, num_columns)
        rowvectors[i] = m * rowvectors[i]
        return cls(rowvectors)

    @classmethod
    def make_row_addition(
        cls, num_rows: int, num_columns: int, i: int, j: int, m: t.Union[int, float]
    ):
        rowvectors = _make_identity_rowvectors(num_rows, num_columns)
        rowvectors[j] = rowvectors[j] + Vector(
            m if j_ == i else 0 for j_ in range(num_columns)
        )
        return cls(rowvectors)

    def __init__(self, rowvectors: t.Iterable[t.Union[Vector, t.Iterable[t.Union[int, float]]]]):
        self._rowvectors = [
            Vector(row) if not isinstance(row, Vector) else row for row in rowvectors
        ]
        _validate_vector_dimensions(self._rowvectors)

    @property
    def columnvectors(self) -> t.Tuple[Vector, ...]:
        return tuple(self.itercolumns())

    @property
    def rowvectors(self) -> t.Tuple[Vector, ...]:
        return tuple(copy.deepcopy(self._rowvectors))

    @property
    def num_columns(self) -> int:
        ## homework:start
        return
        ## homework:end

    @property
    def num_rows(self) -> int:
        ## homework:start
        return
        ## homework:end

    @property
    def shape(self) -> t.Tuple[int, int]:
        return (self.num_rows, self.num_columns)

    @property
    def T(self) -> Matrix:
        if not hasattr(self, "_T"):
            ## homework:start
            self._T = 
            ## homework:end
        return self._T

    @property
    def determinant(self) -> float:
        if not hasattr(self, "_det"):
            ## homework:start
            self._det = 
            ## homework:end
        return self._det

    @property
    def inverse(self) -> Matrix:
        if not hasattr(self, "_inverse"):
            ## homework:start
            self._inverse = 
            ## homework:end
        return self._inverse

    @property
    def trace(self) -> float:
        if not hasattr(self, "_trace"):
            ## homework:start
            self_trace = 
            ## homework:end
        return self._trace

    def iterrows(self) -> t.Generator[Vector, None, None]:
        for row in self._rowvectors:
            yield copy.copy(row)

    def itercolumns(self) -> t.Generator[Vector, None, None]:
        for j in range(self.num_columns):
            yield Vector(self._rowvectors[i][j] for i in range(self.num_rows))

    def __eq__(self, other):
        return almost_equal(self, other)

    def __matmul__(self, other: Matrix) -> Matrix:
        ## homework:start
        return 
        ## homework:end

    def __add__(self, other: Matrix) -> Matrix:
        ## homework:start
        return
        ## homework:end

    def __rmul__(self, k: t.Union[int, float]) -> Matrix:
        ## homework:start
        return
        ## homework:end

    def __neg__(self) -> Matrix:
        ## homework:start
        return
        ## homework:end

    def __sub__(self, other: Matrix) -> Matrix:
        ## homework:start
        return
        ## homework:end

    def __iter__(self):
        return self.iterrows()

    def __len__(self) -> int:
        return self.num_rows

    def __getitem__(self, slice_):
        if isinstance(slice_, int):
            return self._rowvectors[slice_]
        elif isinstance(slice_, slice):
            rowvectors = self._rowvectors[slice_]
            return type(self)(rowvectors)
        elif isinstance(slice_, tuple):
            row, col = slice_
            if isinstance(row, int) and isinstance(col, int):
                return self._rowvectors[row][col]
            elif isinstance(row, slice) and isinstance(col, int):
                return Vector(v[col] for v in self.iterrows())
            elif isinstance(row, int) and isinstance(col, slice):
                return Vector(self._rowvectors[row][col])
            else:
                start, stop, step = self._read_slice(row, self.num_rows)
                rowvectors = (
                    self._rowvectors[i][col] for i in range(start, stop, step)
                )
                return type(self)(rowvectors)
        else:
            raise RuntimeError(f"unsupported slice type {slice_}")

    def __setitem__(self, slice_, value):
        sequence_types = (list, tuple, array, Vector)
        if isinstance(value, sequence_types):
            if len(value) != self.num_columns and len(value != self.num_rows):
                msg = (
                    "vector has inconsisten dimension, must be number of rows ({}) or "
                    "number of columns ({})"
                )
                raise ValueError(msg.format(self.num_rows, self.num_columns))
        error_msg = f"unsoported combination of slice ({slice_}) and values ({value})"
        if isinstance(slice_, int):
            self._rowvectors[slice_][:] = value
        elif isinstance(slice_, slice):
            start, stop, step = self._read_slice(slice_, self.num_rows)
            for i in range(start, stop, step):
                self._rowvectors[i][:] = value
        elif isinstance(slice_, tuple):
            row, col = slice_
            if (
                isinstance(row, (int, slice))
                and isinstance(col, (int, slice))
                and isinstance(value, (int, float, *sequence_types))
            ):
                start_row, stop_row, step_row = self._read_slice(row, self.num_rows)
                start_col, stop_col, step_col = self._read_slice(col, self.num_columns)

                for i in range(start_row, stop_row, step_row):
                    for j in range(start_col, stop_col, step_col):
                        if isinstance(value, sequence_types):
                            v = value[j]
                        else:
                            v = value
                        self._rowvectors[i][j] = v

            else:
                raise TypeError(error_msg)
        else:
            raise TypeError(error_msg)

    def _read_slice(self, slice_, max_stop):
        if isinstance(slice_, int):
            start, stop, step = slice_, slice_ + 1, 1
        else:
            start = 0 if slice_.start is None else slice_.start
            stop = self.num_rows if slice_.stop is None else min(slice_.stop, max_stop)
            step = 1 if slice_.step is None else slice_.step
        return start, stop, step

    def __repr__(self):
        index = len("Vector(")
        vals = "\n  ".join(repr(v)[index:-1] for v in self.iterrows())
        return f"Matrix(\n  {vals[:-1]}],\n shape={self.shape}\n)"


def _validate_vector_dimensions(vectors: t.Sequence[Vector]) -> None:
    ref = vectors[0].dim
    if not all(v.dim == ref for v in vectors):
        raise ValueError(
            "vectors do not have the same t.Union[int, float] of dimensions"
        )


def _make_identity_rowvectors(num_rows, num_columns):
    rowvectors = []
    for i in range(num_rows):
        components = [0] * num_columns
        if i < num_columns:
            components[i] = 1
        rowvectors.append(Vector(components))
    return rowvectors


def scale(m: Matrix, k: t.Union[int, float]) -> Matrix:
    """Scale matrix m by k. """
    ## homework:start
    output_matrix = 
    ## homework:end
    return output_matrix


def add(m1: Matrix, m2: Matrix) -> Matrix:
    """Adds two matrices. """
    ## homework:start
    output_matrix = 
    ## homework:end
    return output_matrix


def subtract(m1: Matrix, m2: Matrix) -> Matrix:
    """Substracts the second matrix from the first one. """
    ## homework:start
    output_matrix = 
    ## homework:end
    return output_matrix


def vector_multiply(m: Matrix, v: Vector, from_left: bool = False) -> Vector:
    """Multiplies a matrix with a vector from the right or the left. """
    cond1 = m.num_rows != v.dim and from_left
    cond2 = m.num_columns != v.dim and not from_left
    if cond1 or cond2:
        raise ValueError(f"Shape mismatch: m({m.shape}), v({v.dim})")

    ## homework:start
    output_vector = 
    ## homework:end
    return output_vector


def matrix_multiply(m1: Matrix, m2: Matrix) -> Matrix:
    """Multiplies two matrices together.

    Args:
        m1 (Matrix): Matrx of shape (m,n)
        m2 (Matrix): Matrix of shape (n, k)

    Returns:
        (Matrix): The product of m1 and m2, has shape (m, k)

    Raises:
        ValueError: if the number of columns in m1 does not match the
        number of rows in m2
    """
    if m1.num_columns != m2.num_rows:
        msg = (
            "num_columns of m1 must equal to num_rows m2, got {} and {} "
            "instead"
        )
        raise ValueError(msg.format(m1.num_columns, m2.num_rows))
    ## homework:start
    output_matrix = 
    ## homework:end
    return output_matrix


def almost_equal(m1: Matrix, m2: Matrix, ndigits: int = PRECISION) -> bool:
    return all(
        vector_ops.almost_equal(v1, v2, ndigits=ndigits) for v1, v2 in zip(m1, m2)
    )


def _validate_matrices_same_dimension(m1: Matrix, m2: Matrix):
    if m1.shape != m2.shape:
        raise ValueError(
            f"matrices must have equal shape, got {m1.shape} and {m2.shape}"
        )


def gaussian_elimination(mat: Matrix) -> Matrix:
    raise NotImplementedError


def lu_decomposition(mat: Matrix) -> Matrix:
    raise NotImplementedError


def eigenvalues(mat: Matrix) -> t.List[t.Union[int, float]]:
    raise NotImplementedError


def eigenvectors(mat: Matrix) -> t.List[Vector]:
    raise NotImplementedError
