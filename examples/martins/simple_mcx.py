'''
Name:         Simple Multicomplex Example of the Martin's function

Description:  This example shows how to compute high-order derivatives of a
              single-variable composite function.

              The function was taken from "The Complex-Step Derivative
              Approximation", by Martins, Sturdza and Alonso (2003).
              https://doi.org/10.1145/838250.838251

Author:       Andres M. Aguirre-Mesa (gch857@my.utsa.edu)
              The University of Texas at San Antonio
Date:         September 9, 2018
Updated:      December 3, 2020
'''


# Libraries and configuration
# ===========================

import os
import sys

# Relative path of the MultiZ library
relpath_multiz = '../../'

# Add MultiZ to Python's path.
cwd = os.path.dirname(os.path.abspath(__file__)) + '/'
sys.path.append(os.path.abspath(cwd + relpath_multiz))

# Import MultiZ
from multiZ.mcomplex import *


# User parameters
# ===============

x0 = 0.5    # Real-valued evaluation point.
h = 1.e-10  # Step size.


# Main script
# ===========

# Define multicomplex input for function.
x = x0 + h*(im(1) + im(2) + im(3) + im(4) + im(5))

# Evaluate multicomplex function.
y = exp(x)/sqrt(sin(x)**3 + cos(x)**3)

# Note: the functions exp, sqrt, sin and cos were defined by MultiZ.

# Compute derivatives from the imaginary parts.
d1 = y.imag(1)/h
d2 = y.imag([1, 2])/h**2
d3 = y.imag([1, 2, 3])/h**3
d4 = y.imag([1, 2, 3, 4])/h**4
d5 = y.imag([1, 2, 3, 4, 5])/h**5

# Note: the comments below show the output for x0 = 0.5 and h = 1.e-10.

print('d1 =', d1)   # d1 = 2.454038334454849
print('d2 =', d2)   # d2 = 2.3559293755346857
print('d3 =', d3)   # d3 = -9.331910038198702
print('d4 =', d4)   # d4 = -55.73181192849724
print('d5 =', d5)   # d5 = 70.32349912943523
