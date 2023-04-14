# MultiZ. Hypercomplex Differentiation Library.
# Copyright (C) 2020 The University of Texas at San Antonio
# 
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License version 2.1 as published by
# the Free Software Foundation.
# 
# This library is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License along
# with this library; if not, write to the Free Software Foundation, Inc., 59
# Temple Place, Suite 330, Boston, MA 

'''
Module name:  symdual.

Description:  This module defines classes and functions to handle symbolic dual
              numbers and arrays.

Date:         January 4, 2021
Updated:      March 16, 2021
'''

from sympy import sympify, simplify

import sympy as sym


class sdual(sym.Expr):

    # Operator priority is set greater than most algebraic expressions (10.0)
    # but lower than Matrix's operator priority (10.01), so an sdual can be
    # contained inside a Matrix (and not the opposite way), and matrix
    # operations are processed correctly by SymPy.
    _op_priority = 10.009

    def __init__(self, real, imag):
        '''
        Constructor of the symbolic dual class.
        '''

        if isinstance(real, type(self)):

            raise TypeError("The value of the real part must be either a "
                            "real-valued symbolics or a numeric expression.")

        elif isinstance(imag, type(self)):

            raise TypeError("The value of the imaginary part must be either "
                            "a real-valued symbolics or numeric expression.")

        else:

            self.real = sympify(real)
            self.imag = sympify(imag)

        # End if. Type cases.

    def __repr__(self):
        '''
        Representation of a symbolic dual object that could be used to create a
        new object.
        '''

        head = self.__class__.__name__ + '('
        body = f'{self.real}, {self.imag}'
        tail = ')'

        return head + body + tail

    def __neg__(self):
        '''
        Negative of a symbolic dual numbers.
        '''

        new = self.__class__(-self.real, -self.imag)

        return new

    def __add__(self, other):
        '''
        Addition of two symbolic dual numbers.
        '''

        a0, a1 = (self.real, self.imag)

        if isinstance(other, type(self)):
            b0, b1 = (other.real, other.imag)
            real = simplify(a0 + b0)
            imag = simplify(a1 + b1)
        else:
            real = simplify(a0 + other)
            imag = a1
        # End if.

        new = self.__class__(real, imag)

        return new

    def __radd__(self, other):
        '''
        Addition of a real number with a dual number.
        '''

        a0, a1 = (self.real, self.imag)

        real = simplify(a0 + other)
        imag = a1

        new = self.__class__(real, imag)

        return new

    def __sub__(self, other):
        '''
        Subtraction of two symbolic dual numbers.
        '''

        a0, a1 = (self.real, self.imag)

        if isinstance(other, type(self)):
            b0, b1 = (other.real, other.imag)
            real = simplify(a0 - b0)
            imag = simplify(a1 - b1)
        else:
            real = simplify(a0 - other)
            imag = a1
        # End if.

        new = self.__class__(real, imag)

        return new

    def __rsub__(self, other):
        '''
        Subtraction of a real number with a dual number.
        '''

        a0, a1 = (self.real, self.imag)

        real = simplify(-a0 + other)
        imag = -a1

        new = self.__class__(real, imag)

        return new

    def __mul__(self, other):
        '''
        Multiplication of two symbolic dual numbers.
        '''

        a0, a1 = (self.real, self.imag)

        if isinstance(other, type(self)):
            b0, b1 = (other.real, other.imag)
            real = simplify(a0 * b0)
            imag = simplify(a0 * b1 + a1 * b0)
        else:
            real = simplify(a0 * other)
            imag = simplify(a1 * other)
        # End if.

        new = self.__class__(real, imag)

        return new

    def __rmul__(self, other):
        '''
        Multiplication of a symbol with a dual number.
        '''

        a0, a1 = (self.real, self.imag)

        real = simplify(a0 * other)
        imag = simplify(a1 * other)

        new = self.__class__(real, imag)

        return new

    def __truediv__(self, other):
        '''
        Division between two dual numbers.
        '''

        a0, a1 = (self.real, self.imag)

        if isinstance(other, type(self)):
            b0, b1 = (other.real, other.imag)
            real = simplify(a0 / b0)
            imag = simplify(1 / b0**2 * (a1 * b0 - a0 * b1))
        else:
            real = simplify(a0 / other)
            imag = simplify(a1 / other)
        # End if.

        new = self.__class__(real, imag)

        return new

    def __rtruediv__(self, other):
        '''
        Division of a real and a dual number.
        '''

        a0, a1 = (self.real, self.imag)
        real = simplify(other / a0)
        imag = simplify(-other * a1 / a0**2)

        new = self.__class__(real, imag)

        return new

    def __pow__(self, other):
        '''
        Power of a dual number raised to another dual number.
        '''

        a0, a1 = (self.real, self.imag)

        if isinstance(other, type(self)):
            b0, b1 = (other.real, other.imag)
            real = simplify(a0**b0)
            imag = simplify((b1 * sym.log(a0) + a1 * b0 / a0) * real)
        else:
            real = simplify(a0**other)
            imag = simplify(other * a0**(other - 1) * a1)
        # End if.

        new = self.__class__(real, imag)

        return new

    def __rpow__(self, other):
        '''
        Power of a real number raised to a dual number.
        '''

        real = simplify(other**self.real)
        imag = simplify((self.imag * sym.log(other)) * real)

        new = self.__class__(real, imag)

        return new

    def as_real_imag(self, deep=True, **hints):

        return (self.real, self.imag)


# End class sdual


def exp(x):
    '''
    Exponential of a sdual number.
    '''

    if isinstance(x, sdual):
        real = simplify(sym.exp(x.real))
        imag = simplify(x.imag * sym.exp(x.real))

        new = sdual(real, imag)
    else:
        new = simplify(sym.exp(x))
    # End if.

    return new


def log(x):
    '''
    Natural logarithm of a sdual number.
    '''

    if isinstance(x, sdual):
        real = simplify(sym.log(x.real))
        imag = simplify(x.imag / x.real)

        new = sdual(real, imag)
    else:
        new = simplify(sym.log(x))
    # End if.

    return new


def sin(x):
    '''
    Sine of a sdual number.
    '''

    if isinstance(x, sdual):
        real = simplify(sym.sin(x.real))
        imag = simplify(x.imag * sym.cos(x.real))

        new = sdual(real, imag)
    else:
        new = simplify(sym.sin(x))
    # End if.

    return new


def cos(x):
    '''
    Cosine of a sdual number.
    '''

    if isinstance(x, sdual):
        real = simplify(sym.cos(x.real))
        imag = simplify(-x.imag * sym.sin(x.real))

        new = sdual(real, imag)
    else:
        new = simplify(sym.cos(x))
    # End if.

    return new


def sqrt(x):
    '''
    Square root of a sdual number.
    '''

    if isinstance(x, sdual):
        real = simplify(sym.sqrt(x.real))
        imag = simplify(x.imag / (2*sym.sqrt(x.real)))

        new = sdual(real, imag)
    else:
        new = simplify(sym.sqrt(x))
    # End if.

    return new


def asin(x):
    '''
    Arcsine of a sdual number.
    '''

    if isinstance(x, sdual):
        real = simplify(sym.asin(x.real))
        imag = simplify(x.imag / sym.sqrt(1 - x.real**2))

        new = sdual(real, imag)
    else:
        new = simplify(sym.asin(x))
    # End if.

    return new


def acos(x):
    '''
    Arccosine of a sdual number.
    '''

    if isinstance(x, sdual):
        real = simplify(sym.acos(x.real))
        imag = simplify(-x.imag / sym.sqrt(1 - x.real**2))

        new = sdual(real, imag)
    else:
        new = simplify(sym.acos(x))
    # End if.

    return new
