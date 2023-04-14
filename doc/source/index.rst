.. MultiZ_py documentation master file, created by
   sphinx-quickstart on Thu Mar 25 16:16:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MultiZ
======

Multicomplex and multidual numbers are two generalizations of complex numbers 
with multiple imaginary axes, useful for numerical computation of derivatives 
with machine precision. The similarities between multicomplex and multidual 
algebras allowed us to create a unified library to use either one for sensitivity 
analysis. This library can be used to compute arbitrary order derivates of functions 
of a single variable or multiple variables. The storage of matrix representations 
of multicomplex and multidual numbers is avoided using a combination of 
one-dimensional resizable arrays and an indexation method based on binary bitwise
operations. To provide high computational efficiency and low memory usage, the 
multiplication of hypercomplex numbers up to sixth order is carried out using a 
hard-coded algorithm. For higher hypercomplex orders, the library uses by default 
a multiplication method based on binary bitwise operations. The computation of 
algebraic and transcendental functions is achieved using a Taylor series approximation. 
Fortran and Python versions were developed, and extensions to other languages are 
self-evident.
   

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

Publications
============

For a more in depth look into the application of multicomplex and multidual
numbers in MultiZ, and to compare documentation to our literature, consider
the following paper:

`MultiZ: A Library for Computation of High-order Derivatives Using Multicomplex
or Multidual Numbers <https://dl.acm.org/doi/10.1145/3378538>`_

Installation
============

MultiZ can be installed or used locally.

Before using the installer, MultiZ must be downloaded. Once downloaded,
you should see MultiZ in your Downloads folder.

If you are an Anaconda Python user, open the Anaconda command prompt. If you
use Python without Anaconda, use your desired terminal. 

On Linux, MacOS (with or without Anaconda), and Windows with MinGW/Cygwin/etc., type the following:

.. code-block :: bash

   $ pip install ~/Downloads/MultiZ_py/
..

or if you don't have admin access, install the package to your user with

.. code-block :: bash
  
   $ pip install ~/Downloads/MultiZ_py/ --user
..

On Windows with Anaconda, type the following:

.. code-block :: bat

   $ pip install ~\Downlods\MultiZ_py\
..

or if you don't have admin access, install the package to your user with

.. code-block :: bat

   $ pip install ~\Downlods\MultiZ_py\ --user
..

To uninstall, use the same command that was used for installation, but replace
"install" with "uninstall".

Once that has been done, simply create a Python script at any location within
your system. Then, you can import MultiZ using one of the following statements

.. code-block :: Python

   from multiZ.mcomplex import *   # To use multicomplex algebra
..

.. code-block :: Python

   from multiZ.mdual import *      # To use multidual algebra.
..

To use MultiZ locally, simply put a copy of the MultiZ folder (located inside this
same folder) inside the same folder in which you are creating your script. Then
you can import MultiZ using one of the following statements

.. code-block :: Python

   from multiZ.mcomplex import *   # To use multicomplex algebra
..

.. code-block :: Python

   from multiZ.mdual import *      # To use multidual algebra.
..
 
To avoid having multiple local copies of MultiZ, you can move a copy of MultiZ
within Python's the default Python's search path. The default search path is
installation dependent. However, you can find it out by entering the following
commands in your Python shell:

.. code-block :: Python
  
   import sys
   sys.path
.. 

The output of these commands is a list of locations where Python looks for
packages and libraries to import.

Multicomplex API
================

.. autoclass :: mcomplex.mcomplex
   :members:
   :special-members: __str__, __mul__, __truediv__, __rtruediv__, __pow__, __rpow__

.. autoclass :: mcomplex.marray
   :special-members: __str__, __mul__, __truediv__, __rtruediv__, __pow__

Multidual API
=============
.. autoclass :: mdual.mdual
   :members:
   :special-members: __str__, __mul__, __truediv__, __rtruediv__, __pow__, __rpow__ 

.. autoclass :: mdual.marray
   :special-members: __str__, __mul__, __truediv__, __rtruediv__, __pow__
