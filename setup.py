import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multiZ",
    version="0.0.3",
    packages=['multiZ'],
    description="MultiZ is a library for computation of high order derivatives using multicomplex or multidual numbers",
    long_description="MultiZ is a library for computation of high order derivatives using multicomplex or multidual numbers",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
