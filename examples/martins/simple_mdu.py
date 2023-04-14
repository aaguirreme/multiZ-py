'''
Name:         Simple Multidual Example of the Martin's function

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
from multiZ.mdual import *


# User parameters
# ===============

x0 = 0.5    # Real-valued evaluation point.
h = 1.      # Step size.


# Main script
# ===========

# Define multidual input for function.
x = x0 + h*(eps(1) + eps(2) + eps(3) + eps(4) + eps(5))

# Evaluate multidual function.
y = exp(x)/sqrt(sin(x)**3 + cos(x)**3)

# Note: the functions exp, sqrt, sin and cos were defined by MultiZ.

# Compute derivatives from the imaginary parts.
d1 = y.imag(1)/h
d2 = y.imag([1, 2])/h**2
d3 = y.imag([1, 2, 3])/h**3
d4 = y.imag([1, 2, 3, 4])/h**4
d5 = y.imag([1, 2, 3, 4, 5])/h**5

# Note: the comments below show the output for x0 = 0.5 and h = 1.

print('d1 =', d1)   # d1 = 2.4540383344548484
print('d2 =', d2)   # d2 = 2.355929375534691
print('d3 =', d3)   # d3 = -9.331910038198695
print('d4 =', d4)   # d4 = -55.73181192849725
print('d5 =', d5)   # d5 = 70.32349912943522
