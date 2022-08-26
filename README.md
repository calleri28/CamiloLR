# Linear Algebra Workbook

**This repository is not intended to be used as a linear algebra library**. Rather, the purpose is to be used as a way to study linear algebra and the common operations defined in linear algebra. You can think of this repository as a self guided workbook for linear algebra.

This repository can also be seen as a Python workbook, since you would get familiar with building a Python library as you go through. This is the reason to not use Numpy (the most used numeric Python library). If we were to use that library, we would end up calling already made functions and objects instead of building our own and understanding the operations used in linear algebra. 

## Your Task

What you should do is that you should implement the library. The library consist of the class vector and matrix, and common operations. You will notice that you are given the test cases for each of the implementaions required. Please report an issue if you find that a test case is wrong. To start, please do:

```bash
git clone https://github.com/open-workbooks/linear-algebra.git
cd linear-algebra
pip install -e ".[dev]"
```

Now you can run the tests and start implementing your own linear algebra library!

Here is the current coverage report:

```bash
---------- coverage: platform darwin, python 3.7.7-final-0 -----------
Name                   Stmts   Miss  Cover   Missing
----------------------------------------------------
lac/__init__.py            3      0   100%
lac/matrix.py            195     35    82%   36-38, 44-46, 52-56, 66, 70, 105, 126-131, 135-142, 189, 204, 212, 218-222, 249-251, 263-265, 271, 362-363, 369, 373, 377, 381
lac/testing_utils.py      16      0   100%
lac/vector.py            115     14    88%   87, 108, 128, 146-147, 163-164, 190-191, 218-219, 252-254, 264
----------------------------------------------------
TOTAL                    329     49    85%

```

### Library desing

We have two simple classes, `Vector` and `Matrix`, which represent the mathematical objects with the same name. These classes offer some ways of building commonly used instances by using classmethods, for instance, the class `Vector` has `classmethod` named `make_random` to build a vector out of random entries. Both classes make some mathematical operator overloading allowed by python, so we can sum two vectors by simply using `+ `. We also implement some features that make the classes look like native Pyhton, by going through the implementation of such classes you could learn a lot about Python.

We reserve the use of the `property` decorator to attributes that are of the very nature of the object. For instance, every vector has a norm (here we use the euclidean one) and that is a property of each `Vector` instance. The same is true for the `determinant` and `inverse` of a matrix, even though they may be undefined, which is reported by raising an error with a well descriptive message. Additionally, there are some functions that operate on vectors, some on matrices and some on both. 

The tests cases for all implementations aim to hide implementation details, since otherwise the implementation can be derived from there. We make an effort to implement tests cases as the mathematical properties of whatever is being tested. Hopefully this will give the user yet another chance to see how mathematical relations can be implemented in Python code.

Finally, we favor type hints and self descriptive names over docstrings, but we make use of docstrings when we need to (when it is not redundant given the names and typehints used.)

### How you use your library

Here is an example of the things you can do with the `Vector` instances (once you implement the `Vector` class):

```python
import lac

v1 = lac.Vector([1, 2, 3])
v2 = lac.Vector([4, 5, 6])

v3 = v1 + v2 
print(v3) # Vector([5.0, 7.0, 9.0])
print(v3.norm) # 12.449899597988733

v4 = lac.vector.cross(v1, v2)
print(v4) # Vector([-3.0, 6.0, -3.0])

alpha = lac.vector.angle_between(v1, v3)
print(alpha) # 1.5707963267948966
```

An here is how the `Matrix` instances could be used:

```python
import lac

m1 = lac.Matrix([[1,2,3],[4,5,6],[7,8,9]])
m2 = m1.T
print(m2)
# Matrix(
#   [1.0, 4.0, 7.0]
#   [2.0, 5.0, 8.0]
#   [3.0, 6.0, 9.0],
#  shape=(3, 3)
# )

m3 = m1 @ m2
print(m3)
# Matrix(
#   [14.0, 32.0, 50.0]
#   [32.0, 77.0, 122.0]
#   [50.0, 122.0, 194.0],
#  shape=(3, 3)
# )
```

## Where to get the answers

The workbook is splited into two repositories `open-workbooks/linear-algebra` and `open-workbooks/linear-algebra-answers`, which hold the task and the answers respectively. Since this worbook can be used as teaching material, the repo with the ansers is a private repository, but we can give you access to the private repository if you fill out our [collaborators form](https://forms.gle/atFNQEUxryN72L189). The repo with the answers has CI/CD implemented that so that all pushes to master update the repo without the answers, we can make sure that the task repo is always updated.

## Contribute

All forms of contributing are highly appreciated, please read the [contributing guide](./CONTRIBUTING.md)

## Collaborators

- Sebastián Rodríguez Colina
- [Add your name here!](./CONTRIBUTING.md)

## References

You could see this workbook is not intended to be a standalone linear algebra reference. Here are some refences we have found to be extremely helpfull to study the subject:

- [Introduction to Linear Algebra by Gilbert Strang (2016)](https://math.mit.edu/~gs/linearalgebra/)
- [MIT A 2020 Vision of Linear Algebra, Spring 2020](https://www.youtube.com/playlist?list=PLUl4u3cNGP61iQEFiWLE21EJCxwmWvvek)
- [MIT 18.06SC Linear Algebra, Fall 2011](https://www.youtube.com/playlist?list=PL221E2BBF13BECF6C)
- [No Bullshit Guide to Linear Algebra by Ivan Savov](https://www.goodreads.com/book/show/34760208-no-bullshit-guide-to-linear-algebra)
- [Essence of linear algebra by 3Bue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
