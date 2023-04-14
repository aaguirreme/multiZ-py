# multiZ-py
MultiZ is a multicomplex and multidual algebra library. It is useful for developing or modifying code to calculate arbitrary order derivatives.

Requirements
============

- Python or Anaconda Python 3.5.0 or higher.
- NumPy 1.18.0 or higher.
- To run the interactive tutorial notebooks, located in the tutorials folder,
  you need Jupyter 1.0.0 or higher.

Installation
============

MultiZ can be installed or used locally.

Before using the installer, MultiZ must be downloaded. Once downloaded,
you should see MultiZ in your Downloads folder.

If you are an Anaconda Python user, open the Anaconda command prompt. If you
use Python without Anaconda, use your desired terminal. 

On Linux, MacOS (with or without Anaconda), and Windows with MinGW/Cygwin, type the following:

```
$ pip install ~/Downloads/MultiZ_py/
```

or if you don't have admin access, install the package to your user with

```
$ pip install ~/Downloads/MultiZ_py/ --user
```

On Windows with Anaconda, type the following:

```
$ pip install ~\Downlods\MultiZ_py\
```

or if you don't have admin access, install the package to your user with

```
$ pip install ~\Downlods\MultiZ_py\ --user
```

To uninstall, use the same command that was used for installation, but replace
"install" with "uninstall".

Once that has been done, simply create a Python script at any location within
your system. Then, you can import MultiZ using one of the following statements

```
from multiZ.mcomplex import *   # To use multicomplex algebra
```

```
from multiZ.mdual import *      # To use multidual algebra.
```

To use MultiZ locally, simply put a copy of the MultiZ folder (located inside this
same folder) inside the same folder in which you are creating your script. Then
you can import MultiZ using one of the following statements

```
from multiZ.mcomplex import *   # To use multicomplex algebra
```

```
from multiZ.mdual import *      # To use multidual algebra.
```

To avoid having multiple local copies of MultiZ, you can move a copy of MultiZ
within Python's the default Python's search path. The default search path is
installation dependent. However, you can find it out by entering the following
commands in your Python shell:

```
import sys
sys.path
```

The output of these commands is a list of locations where Python looks for
packages and libraries to import.

Tutorials and Examples
======================

The tutorials folder contains multiple Jupyter notebooks that cover basic usage
and the mathematical features of the multicomplex and multidual algebra
implementations of MultiZ.

The examples folder contains multiple Python scripts that show how to use
multicomplex and multidual algebra for the approximation of derivatives of
real-valued functions.
