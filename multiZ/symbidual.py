# MultiZ. Hypercomplex Differentiation Library.
# Copyright (C) 2020 The University of Texas at San Antonio
# 
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License version 2.1 as published
# by the Free Software Foundation.
# 
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 

'''
Module name:  symbidual.

Description:  This module defines classes and functions to handle symbolic
              bidual numbers and arrays.

Date:         January 4, 2021
Updated:      March 16, 2021
'''

from sympy import sympify, simplify, S

import sympy as sym


class sbdual(sym.Expr):

    # Operator priority is set greater than most algebraic expressions (10.0)
    # but lower than Matrix's operator priority (10.01), so an sbdual can be
    # contained inside a Matrix (and not the opposite way), and matrix
    # operations are processed correctly by SymPy.
    _op_priority = 10.009

    def __init__(self, real, imag1, imag2, imag12):
        '''
        Constructor of the symbolic bidual class.
        '''

        imag_i = [imag1, imag2, imag12]

        for imag_part in imag_i:

            if isinstance(imag_part, type(self)):

                raise TypeError("Imaginary part values must be real-value "
                                "symbolics or numeric expressions")

            else:

                self.real = sympify(real)
                self.imag1 = sympify(imag1)
                self.imag2 = sympify(imag2)
                self.imag12 = sympify(imag12)

    def __repr__(self):
        '''
        Representation of a symbolic bidual object that could be used to create
        a new object.
        '''

        a0, a1, a2, a12 = (self.real, self.imag1, self.imag2, self.imag12)

        head = self.__class__.__name__ + '('
        body = f'{a0}, {a1}, {a2}, {a12}'
        tail = ')'

        return head + body + tail

    def __neg__(self):
        '''
        Negative of a symbolic bidual numbers.
        '''

        a0, a1, a2, a12 = (self.real, self.imag1, self.imag2, self.imag12)

        new = self.__class__(-a0, -a1, -a2, -a12)

        return new

    def __add__(self, other):
        '''
        Addition of two symbolic bidual numbers.
        '''

        a0, a1, a2, a12 = (self.real, self.imag1, self.imag2, self.imag12)

        if isinstance(other, type(self)):
            b0, b1, b2, b12 = (other.real, other.imag1,
                               other.imag2, other.imag12)
            real = simplify(a0 + b0)
            imag1 = simplify(a1 + b1)
            imag2 = simplify(a2 + b2)
            imag12 = simplify(a12 + b12)
        else:
            real = simplify(a0 + other)
            imag1 = a1
            imag2 = a2
            imag12 = a12
        # End if.

        new = self.__class__(real, imag1, imag2, imag12)

        return new

    def __radd__(self, other):
        '''
        Addition of a real number with a bidual number.
        '''

        a0, a1, a2, a12 = (self.real, self.imag1, self.imag2, self.imag12)

        real = simplify(a0 + other)
        imag1 = a1
        imag2 = a2
        imag12 = a12

        new = self.__class__(real, imag1, imag2, imag12)

        return new

    def __sub__(self, other):
        '''
        Subtraction of two symbolic bidual numbers.
        '''

        a0, a1, a2, a12 = (self.real, self.imag1, self.imag2, self.imag12)

        if isinstance(other, type(self)):
            b0, b1, b2, b12 = (other.real, other.imag1,
                               other.imag2, other.imag12)
            real = simplify(a0 - b0)
            imag1 = simplify(a1 - b1)
            imag2 = simplify(a2 - b2)
            imag12 = simplify(a12 - b12)
        else:
            real = simplify(a0 - other)
            imag1 = a1
            imag2 = a2
            imag12 = a12
        # End if.

        new = self.__class__(real, imag1, imag2, imag12)

        return new

    def __rsub__(self, other):
        '''
        Subtraction of a real number with a bidual number.
        '''

        a0, a1, a2, a12 = (self.real, self.imag1, self.imag2, self.imag12)

        real = simplify(-a0 + other)
        imag1 = -a1
        imag2 = -a2
        imag12 = -a12

        new = self.__class__(real, imag1, imag2, imag12)

        return new

    def __mul__(self, other):
        '''
        Multiplication of two symbolic bidual numbers.
        '''

        a0, a1, a2, a12 = (self.real, self.imag1, self.imag2, self.imag12)

        if isinstance(other, type(self)):
            b0, b1, b2, b12 = (other.real, other.imag1,
                               other.imag2, other.imag12)
            real = simplify(a0 * b0)
            imag1 = simplify(a0 * b1 + a1 * b0)
            imag2 = simplify(a0 * b2 + a2 * b0)
            imag12 = simplify(a0 * b12 + a1 * b2 + a2 * b1 + a12 * b0)
        else:
            real = simplify(a0 * other)
            imag1 = simplify(a1 * other)
            imag2 = simplify(a2 * other)
            imag12 = simplify(a12 * other)
        # End if.

        new = self.__class__(real, imag1, imag2, imag12)

        return new

    def __rmul__(self, other):
        '''
        Multiplication of a symbol with a dual number.
        '''

        a0, a1, a2, a12 = (self.real, self.imag1, self.imag2, self.imag12)

        real = simplify(a0 * other)
        imag1 = simplify(a1 * other)
        imag2 = simplify(a2 * other)
        imag12 = simplify(a12 * other)

        new = self.__class__(real, imag1, imag2, imag12)

        return new

    def __truediv__(self, other):
        '''
        Division of a bidual number with a bidual number or real number.
        '''

        a, b, c, d = (self.real, self.imag1, self.imag2, self.imag12)

        if isinstance(other, type(self)):
            m, n, p, q = (other.real, other.imag1, other.imag2, other.imag12)

            real = simplify(a / m)
            imag1 = simplify((-a*n + b*m)/m**2)
            imag2 = simplify((-a*p + c*m)/m**2)
            imag12 = simplify((-a*(m*q - 2*n*p) + d*m**2 - m*(b*p + c*n))/m**3)
        else:
            m = other

            real = simplify(a / m)
            imag1 = simplify(b / m)
            imag2 = simplify(c / m)
            imag12 = simplify(d / m)

        new = self.__class__(real, imag1, imag2, imag12)
            # End if.
        return new

    def __rtruediv__(self, other):
        '''
        Division of a real and a bidual number.
        '''

        a0, a1, a2, a12 = (self.real, self.imag1, self.imag2, self.imag12)

        real = simplify(other / a0)
        imag1 = simplify(-other * a1 / a0**2)
        imag2 = simplify(-other * a2 / a0**2)
        imag12 = simplify(other * (-a0*a12 + 2*a1*a2) / a0**3)

        new = self.__class__(real, imag1, imag2, imag12)

        return new

    def __pow__(self, other):
        '''
        Power of a dual number raised to another dual number.
        '''

        if isinstance(other, type(self)):
            a, b, c, d = (self.real, self.imag1, self.imag2, self.imag12)
            m, n, p, q = (other.real, other.imag1, other.imag2, other.imag12)

            real = simplify(sym.exp(m*sym.log(a)))
            imag1 = simplify((a*n*sym.log(a) + b*m)*sym.exp(m*sym.log(a))/a)
            imag2 = simplify(((a*p*sym.log(a) + c*m)*sym.exp(m*sym.log(a)))/a)
            imag12 = simplify((a**2*q*sym.log(a) + a*(b*p + c*n) + m*(a*d - b*c) + (a*n*sym.log(a) + b*m)*(a*p*sym.log(a) + c*m))* \
                        exp(m*sym.log(a))/a**2)

            new = self.__class__(real,imag1,imag2,imag12)
        elif sympify(other) == sym.S(1):
            new = self.copy()
        else:
            a0, a1, a2, a12 = (self.real, self.imag1, self.imag2, self.imag12)
            real = simplify(a0**other)
            imag1 = simplify(other * a0**(other - 1) * a1)
            imag2 = simplify(other * a0**(other - 1) * a2)
            imag12 = simplify(other * a0**(other - 1) * a12
                              + other * (other - 1) * a0**(other - 2) * a1 * a2)

            new = self.__class__(real, imag1, imag2, imag12)
        # End if.

        return new

    def __rpow__(self, other):
        '''
        Power of a real number raised to a dual number.
        '''

        a, b, c, d = (self.real, self.imag1, self.imag2, self.imag12)

        real = simplify(sym.exp(a*sym.log(other)))
        imag1 = simplify(b*sym.exp(a*sym.log(other))*sym.log(other))
        imag2 = simplify(c*sym.exp(a*sym.log(other))*sym.log(other))
        imag12 = simplify((b*c*sym.log(other) + d)*sym.exp(a*sym.log(other))*sym.log(other))

        new = self.__class__(real, imag1, imag2, imag12)

        return new

    def as_real_imag(self, deep=True, **hints):

        a0, a1, a2, a12 = (self.real, self.imag1, self.imag2, self.imag12)
        return (a0, a1, a2, a12)

# End class sbdual

def exp(x):
    '''
    Exponential of a sbdual number.
    '''

    if isinstance(x, sbdual):
        
        a, b, c, d = (x.real, x.imag1, x.imag2, x.imag12)

        real = simplify(sym.exp(a))
        imag1 = simplify(b * sym.exp(a))
        imag2 = simplify(c * sym.exp(a))
        imag12 = simplify((b * c + d) * sym.exp(a))

        new = sbdual(real, imag1, imag2, imag12)
    else:
        new = simplify(sym.exp(x))
    # End if.
    
    return new

def log(x):
    '''
    Natural logarithm of a sbdual number.
    '''
    
    if isinstance(x,sbdual):
        a,b,c,d = (x.real, x.imag1, x.imag2, x.imag12)
        
        real = simplify(sym.log(a)) 
        imag1 = simplify(b/a)
        imag2 = simplify(c/a)
        imag12 = simplify((a*d - b*c)/a**2)
        
        new = sbdual(real, imag1,imag2,imag12)
    else:
        new = simplify(sym.log(x))
    # End if.

    return new

def sin(x):
    '''
    Sine of a sbdual number.
    '''

    if isinstance(x, sbdual):
        a, b, c, d = (x.real, x.imag1, x.imag2, x.imag12)

        real = simplify(sym.sin(a))
        imag1 = simplify(b * sym.cos(a))
        imag2 = simplify(c * sym.cos(a))
        imag12 = simplify(-b * c * sym.sin(a) + d * sym.cos(a))

        new = sbdual(real, imag1, imag2, imag12)
    else:
        new = simplify(sym.sin(x))
    # End if.

    return new

def cos(x):
    '''
    Cosine of a sbdual number.
    '''

    if isinstance(x, sbdual):
        a,b,c,d = (x.real, x.imag1, x.imag2, x.imag12)

        real = simplify(sym.cos(a))
        imag1 = simplify(-b*sym.sin(a))
        imag2 = simplify(-c*sym.sin(a))
        imag12 = simplify(-b*c*sym.cos(a) - d*sym.sin(a))

        new = sbdual(real,imag1,imag2,imag12)
    else:
        new = simplify(sym.cos(x))
    # End if.

    return new

def sqrt(x):
    '''
    Square root of a sbdual number.
    '''

    if isinstance(x, sbdual):
        a,b,c,d = (x.real,x.imag1,x.imag2,x.imag12)

        real = simplify(sym.sqrt(a))
        imag1 = simplify(b/(2*sym.sqrt(a)))
        imag2 = simplify(c/(2*sym.sqrt(a)))
        imag12 = simplify(-a**(-S(3)/2)*b*c/4 + a**(-S(1)/2)*d/2)

        new = sbdual(real,imag1,imag2,imag12)
    else:
        new = simplify(sym.sqrt(x))
    # End if.

    return new

def asin(x):
    '''
    Arcsine of a sbdual number.
    '''

    if isinstance(x, sbdual):
        a, b, c, d = (x.real, x.imag1, x.imag2, x.imag12)

        real = simplify(sym.asin(a))
        imag1 = simplify(b/sym.sqrt(1 - a**2))
        imag2 = simplify(c/sym.sqrt(1 - a**2))
        imag12 = simplify(a*b*c*(1 - a**2)**(-S(3)/2) + d*(1 - a**2)**(-S(1)/2))

        new = sbdual(real, imag1, imag2, imag12)
    else:
        new = simplify(sym.asin(x))
    # End if.

    return new

def acos(x):
    '''
    Arccos of a sbdual number.
    '''

    if isinstance(x, sbdual):
        a, b, c, d = (x.real, x.imag1, x.imag2, x.imag12)

        real = simplify(sym.acos(a))
        imag1 = simplify(-b/sym.sqrt(1 - a**2))
        imag2 = simplify(-c/sym.sqrt(1 - a**2))
        imag12 = simplify(-a*b*c*(1 - a**2)**(-S(3)/2) - d*(1 - a**2)**(-S(1)/2))

        new = sbdual(real, imag1, imag2, imag12)
    else:
        new = simplify(sym.acos(x))
    # End if.

    return new
