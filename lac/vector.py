from __future__ import annotations

import math
import random
import reprlib
import typing as t
from array import array
from collections.abc import Iterable

from lac import PRECISION


class Vector:
    typecode = "d"

    @classmethod
    def make_random(cls, dim):
        """Make a unitary vector of random components. """
        return cls(random.random() for _ in range(dim))

    @classmethod
    def make_zero(cls, dim: int):
        """Make a zero vector of any lenght. """
        return cls([0] * dim)

    @classmethod
    def make_unitary(cls, components: t.Iterable[t.Union[int, float]]):
        """Make a unitary vetor out of the components. """
        return build_unit_vector(cls(components))

    def __init__(self, components: t.Iterable[t.Union[int, float]]):
        self._components = array(self.typecode, components)

    @property
    def components(self) -> array:
        return self._components

    @property
    def norm(self) -> float:
        if not hasattr(self, "_norm"):
            ## homework:start
            self._norm = math.sqrt(sum(i**2 for i in self))
            ## homework:end
        else:
            pass
        return self._norm

    @property
    def dim(self) -> int:
        ## homework:start
        cont=0
        for i in self:
            cont+=1
        return cont
        ## homework:end

    def __add__(self, other: Vector) -> Vector:
        ## homework:start
        return add(self, other)
        ## homework:end
    
    def __rmul__(self, k: t.Union[int,float]) -> Vector:
       return scale(self,k)

    def __matmul__(v1:Vector, v2:Vector) -> Vector:
       return dot(v1,v2)

    def __neg__(self) -> Vector:
        ## homework:start
        output_vector = scale(self, -1)
        return output_vector
        ## homework:end

    def __sub__(self, other: Vector) -> Vector:
        ## homework:start
        return subtract(self, other)
        ## homework:end

    def __abs__(self):
        ## homework:start
        return self.norm
        ## homework:end

    def __len__(self):
        ## homework:start
        return self.dim
        ## homework:end

    def __eq__(self, other):
        return almost_equal(self, other)

    def __iter__(self):
        return iter(self.components)

    def __getitem__(self, slice_):
        if isinstance(slice_, int):
            return self.components[slice_]
        elif isinstance(slice_, slice):
            return Vector(self.components[slice_])
        else:
            raise RuntimeError("unsupported slice")

    def __setitem__(self, slice_, value):
        if isinstance(slice_, int) and isinstance(value, (int, float)):
            self.components[slice_] = value
        elif isinstance(slice_, slice) and isinstance(
            value, (int, float, list, tuple, array, Vector)
        ):
            start = 0 if slice_.start is None else slice_.start
            stop = (
                len(self.components)
                if slice_.stop is None
                else min(slice_.stop, len(self.components))
            )
            step = 1 if slice_.step is None else slice_.step
            length = math.ceil((stop - start) / step)
            if isinstance(value, (int, float)):
                value = [value] * length
            self.components[slice_] = array(self.typecode, value)
        else:
            raise TypeError(
                f"unsupported combination of slice ({slice_}) and value ({value})"
            )

    def __repr__(self):
        components = reprlib.repr([round(c, 3) for c in self.components])
        idx = components.find("[")
        components = components[idx:]
        return f"Vector({components})"


def scale(v: Vector, k: t.Union[int, float]) -> Vector:
    """Scales a vector by k.

    Raises:
        TypeError: if k is not a t.Union[int, float].
    """
    if not isinstance(k, (int, float)):
        msg = "Vectors can only be scaled by scalars! got {}"
        raise TypeError(msg.format(type(k)))

    ## homework:start
    output_vector = Vector(e*k for e in v)
    ## homework:end
    return output_vector


def add(v1: Vector, v2: Vector) -> Vector:
    """Adds two vectors of the same dimension.

    Raise:
        ValueError: if vectors do not have the same dimesion.
    """
    if v1.dim != v2.dim:
        msg = "Vectors must have the same dimension, got {} and {}"
        raise ValueError(msg.format(v1.dim, v2.dim))

    ## homework:start
    output_vector = Vector(a+b for a,b in zip(v1,v2))
    ## homework:end
    return output_vector


def subtract(v1: Vector, v2: Vector) -> Vector:
    """Subtracts the second vector from the first vector. """
    ## homework:start
    output_vector = Vector(a-b for a,b in zip(v1,v2))
    ## homework:end
    return output_vector


def dot(v1: Vector, v2: Vector) -> float:
    """Computes the dot product of two vectors.

    Raises:
        ValueError: if vectors do not have the same dimension

    """
    if v1.dim != v2.dim:
        msg = "vectors must have the same dimension, got {} and {}"
        raise ValueError(msg.format(v1.dim, v2.dim))
    ## homework:start
    sum=0
    x = (a*b for a,b in zip(v1,v2))
    for i in x:
        sum += i
    output_value = sum
    ## homework:end
    return output_value


def angle_between(v1: Vector, v2: Vector) -> float:
    """Computes the angle between two vectors. """
    ## homework:start
    aux = dot(v1, v2)/(v1.norm*v2.norm)
    aux = max(min(aux,1),-1)
    alpha = math.acos(aux)
    ## homework:end
    return alpha


def cross(v1: Vector, v2: Vector) -> Vector:
    """Computes the cross product between two 3-dimensional vectors.

    Raises:
        ValueError: if any of the vectors in not 3 dimentional.
    """
    for v in [v1, v2]:
        if v.dim != 3:
            msg = "Expected 3-dimentional vector, got a {}-dimentional"
            raise ValueError(msg.format(v.dim))

    ## homework:start
    aux1=v1[1]*v2[2] - v1[2]*v2[1]
    aux2=v1[0]*v2[2] - v1[2]*v2[0]
    aux3=v1[0]*v2[1] - v1[1]*v2[0]
    output_vector = Vector([aux1,-aux2,aux3])
    ## homework:end
    return output_vector


def build_unit_vector(v: Vector) -> Vector:
    """Builds a unit vector from the provided vector. """
    ## homework:start
    unit_vector = scale(v,(1/v.norm))
    ## homework:end
    return unit_vector


def project(v: Vector, d: Vector) -> Vector:
    """Projects a vector v into the vector d.

    Arguments:
        v (Vector): the vector you wish to project.
        d (Vector): the fector you wish to project onto.

    Returns:
        Vector: the projection of v onto d.
    """
    ## homework:start
    num = dot (v,d)
    den = d.norm**2
    projection_vector = scale(d,(num/dem))
    ## homework:end
    return projection_vector


def almost_equal(v1: Vector, v2: Vector, ndigits: int = PRECISION) -> bool:
    _validate_vectors_same_dim(v1, v2)
    return all(round(c1, ndigits) == round(c2, ndigits) for c1, c2 in zip(v1, v2))


def _validate_vectors_same_dim(v1: Vector, v2: Vector):
    if v1.dim != v2.dim:
        raise ValueError(
            f"vectors must have the same dimension, got {v1.dim} and {v2.dim}"
        )
