from setuptools import setup, find_packages

dev_required = ["pytest", "pytest-xdist", "pytest-cov", "black", "mypy", "pydocstyle"]

setup(
    name="lac",
    version="0.0.1",
    description="A companion to study linear algebra",
    url="https://github.com/open-workbooks/linear-algebra",
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
    extras_require={"dev": dev_required},
    package_dir={"": "."},
)
