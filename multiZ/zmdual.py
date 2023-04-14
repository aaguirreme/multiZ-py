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
Module name:  mdual.

Description:  This is an experimental module to define the subclasses and
              functions of a complex-coefficient multidual algebra, capable of
              higher-order differentiation of complex-variable functions.

Date:         May 5, 2016
Updated:      March 16, 2021
'''

import numpy as np

import multiZ.zcore as core

from multiZ.zcore import getImagDirs, getPos


class mdual(core.mnumber):

    def __str__(self):
        """
        Pretty print representation of a multidual number.

        :param self: A multidual number to be converted into a Pretty print
            prepresentation
        :type self: mdual

        .. code-block:: Python

            >>> print(mdual([1,2,3,4]))
            1 + 2*eps(1) + 3*eps(2) + 4*eps([1, 2])
        """

        return core.multi_str(self, "eps")

    def __mul__(self, other):
        """
        To define how to multiply two multidual numbers.
     
        It overloads the multiplication operator "*". It allows
        the multiplication of two multidual numbers of different
        orders, no matter how they are sorted, or even the
        multiplication of a multidual and a scalar.
     
        :return: mdual
        :param self: multidual number
        :param other: multidual number
        :type self: mdual
        :type other: mdual
     
        The general, symbolic representation for multiplication
        of two symbolic dual vectors

        .. math:: D_1 = sdual\left ( a, b \\right )

        and

        .. math:: D_2 = sdual\left ( c, d \\right )

        can be generally represented as

        .. math:: sdual\left ( ac, ad + bc \\right ).

        .. code-block:: Python

           >>> a = mdual([1,2,3])
           >>> b = mdual([4,5])
           >>> print(a * b)
           -6 + 13*eps(1) + 12*eps(2) + 15*eps([1, 2])

        ..

        And for the case of two symbolic bidual numbers,

        .. math :: D_1 = sbdual\left ( a,b,c,d \\right)

                   D_2 = sbdual\left ( e,f,g,h \\right)

        the multiplication of the two can be generally represented as

        .. math :: sbdual\left (ae, af + be, ag + ce, ah + bg, cf + de \\right) .

        """

        coeffs1 = self.coeffs                   # Get attributes from self.
        order1 = self.order
        n1 = self.numcoeffs

        if type(other) in core.scalartypes:     # Case 1. Mult. by real num.

            new = self.__class__(0)             # Create a new mdual number.

            new.coeffs = coeffs1 * other        # Overwrite attributes.
            new.order = order1
            new.numcoeffs = n1

        elif isinstance(other, type(self)):

            # Case 2. Mult. by mdual.

            new = self.__class__(0)             # Create a new mdual number.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            if (n1 >= n2):

                order = order1                  # Set attributes for result.
                n = n1
                p = np.zeros(n, np.complex128)     # Allocate space

                core.mdual_mul(coeffs1, n1, coeffs2, n2, p)

            else:

                order = order2                  # Set attributes for result.
                n = n2
                p = np.zeros(n, np.complex128)     # Allocate space

                core.mdual_mul(coeffs2, n2, coeffs1, n1, p)

            new.coeffs = p                      # Overwrite attributes.
            new.order = order
            new.numcoeffs = n

        elif isinstance(other, marray):         # Case 3. Mult. by array.

            new = other.__class__([[0]])        # New array for result.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            # Get array attributes from other.
            shape2 = other.shape
            size2 = other.size

            if (n2 >= n1):

                new_coeffs = np.zeros(shape2 + (n2,), np.complex128)
                order = order2
                n = n2

                for i in range(size2):

                    idx = core.item2index(i, shape2, size2)

                    core.mdual_mul(coeffs2[idx], n2,
                                   coeffs1, n1, new_coeffs[idx])

            else:

                new_coeffs = np.zeros(shape2 + (n1,), np.complex128)
                order = order1
                n = n1

                for i in range(size2):

                    idx = core.item2index(i, shape2, size2)

                    core.mdual_mul(
                        coeffs1, n1, coeffs2[idx], n2, new_coeffs[idx])

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order
            new.numcoeffs = n
            new.shape = shape2
            new.size = size2

        else:                                   # Case 4. Incompatible types.

            raise TypeError("unsupported operand type(s) for *: '%s' and '%s'"
                            % (core.strtype(self), core.strtype(other)))

        return new

    def __truediv__(self, other):
        """
        :return: marray

        Provides division of an array of multidual values with scalars and
        multidual numbers.
    
        :param self: multidual number
        :param other: scalar or multidual number to be divided with self
        :type self: marray
        :type other: mdual, int, float

        For the case of two symbolic bidual numbers,

        .. math :: D_1 = sbdual\left ( a,b,c,d \\right)

                   D_2 = sbdual\left ( e,f,g,h \\right)

        the division of the two can be generally represented as

        .. math :: sbdual\left ( \\frac{a}{e}, \\frac{-af + be}{e^{2}}, 
                         \\frac{-ag + ce}{e^{2}}, 
                         \\frac{-a(eh - 2fg) + de^{2} - e(bg + cf)}{e^{3}} \\right ) .
        
        .. code-block :: Python
        
            >>> a = mdual([1,2,3,4])
            >>> b = mdual([5,6,7,8])
            >>> print(a/b)
            0.2 + 0.16*eps(1) + 0.32*eps(2) - 0.128*eps([1, 2])
        ..
        """

        new = self.__class__(0)                 # Create a new mdual number.

        coeffs1 = self.coeffs                   # Get attributes from self.
        order1 = self.order
        n1 = self.numcoeffs

        if type(other) in core.scalartypes:     # Case 1. Div by real number.

            new.coeffs = coeffs1 / other        # Overwrite attributes.
            new.order = order1
            new.numcoeffs = n1

        elif isinstance(other, type(self)):

            # Case 2. Div by multidual.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            # Find max number of coefficients.
            if (n1 >= n2):
                order = order1
                n = n1
            else:
                order = order2
                n = n2

            # Allocate space for coefficients.
            num = np.zeros(n, np.complex128)       # Numerator.
            den = np.zeros(n, np.complex128)       # Denominator.
            old = np.zeros(n, np.complex128)       # Space for copies.
            cden = np.zeros(n, np.complex128)      # Conjugated denominator.

            # Initialize numerator and denominator.
            num[:n1] = coeffs1.copy()
            den[:n2] = coeffs2.copy()

            # Compute coefficients of the numerator.
            core.mdual_truediv(num, den, old, cden, n, order2)

            new.coeffs = num                    # Overwrite attributes.
            new.order = order
            new.numcoeffs = n

        elif isinstance(other, marray):         # Case 3. Division by array.

            new = other.__class__([[0]])        # New array for result.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            # Get array attributes from other.
            shape = other.shape
            size = other.size

            # Find max number of coefficients.
            if (n1 >= n2):
                order = order1
                n = n1
            else:
                order = order2
                n = n2

            # Allocate space for coefficients.
            num = np.zeros(shape + (n,), np.complex128) # Numerator.
            den = np.zeros(n, np.complex128)            # Denominator.
            old = np.zeros(n, np.complex128)            # Space for copies.
            cden = np.zeros(n, np.complex128)           # Conjugated denominator.

            for i in range(size):

                idx = core.item2index(i, shape, size)

                # Initialize numerator and denominator.
                num[idx + (slice(0, n1),)] = coeffs1.copy()
                den[:n2] = coeffs2[idx].copy()

                # Compute coefficients of the numerator.
                core.mdual_truediv(num[idx], den, old, cden, n, order)

            new.coeffs = num                    # Overwrite attributes.
            new.order = order
            new.numcoeffs = n
            new.shape = shape
            new.size = size

        else:                                   # Case 4. Incompatible types.

            raise TypeError("unsupported operand type(s) for /: '%s' and '%s'"
                            % (core.strtype(self), core.strtype(other)))

        return new

    def __rtruediv__(self, other):
        """
        
        To provide reverse division for multidual arrays.

        It defines how to make a division between a scalar and a
        multidual number.

        :param self: Real value
        :param other: multidual number
        :type self: int, float
        :type other: mdual
    
        Let :math:`D` be a symbolic bidual number, where

        .. math :: D = sbdual \\left (a,b,c,d \\right )
            
        and :math:`r` is any real number. The result of :math:`r / D` is

        .. math:: sbdual \\left ( \\frac{r}{a},
                                \\frac{-br}{a^{2}},
                                \\frac{-cr}{a^{2}},
                                \\frac{r(-ad + 2bc)}{a^{3}}
                         \\right ) .
 
        .. code-block :: Python
           
            >>> r = 2
            >>> a = mdual([1,2,3,4])
            >>> print(r/a)    
            2 - 4*eps(1) - 6*eps(2) + 16*eps([1, 2])
        ..    
        """

        new = self.__class__(0)                 # Create a new mdual number.
        coeffs = self.coeffs                    # Create a view to self.coeffs.
        order = self.order                      # Get attributes.
        n = self.numcoeffs

        # Allocate space for coefficients.
        num = np.zeros(n, np.complex128)           # Numerator.
        old = np.zeros(n, np.complex128)           # Space for copies.
        cden = np.zeros(n, np.complex128)          # Conjugated denominator.

        num[0] = other                          # Initialize numerator.

        # Allocate and initialize denominator.
        den = coeffs.copy()

        # Compute coefficients of the numerator.
        core.mdual_truediv(num, den, old, cden, n, order)

        new.coeffs = num                        # Overwrite attributes.
        new.order = order
        new.numcoeffs = n

        return new


    def __pow__(self, other):
        """
        :return: marray

        Provides the power function method for multidual arrays.
        
        This function makes use of the Taylor series expansion approach, which is
        exact for multidual numbers.
        
        :param self: multidual array
        :param e: multidual array to raise the multidual array by
        :type self: marray
        :type e: marray

        Given two symbolic bidual numbers,

        .. math :: D_1 = sbdual\left ( a,b,c,d \\right)

                   D_2 = sbdual\left ( e,f,g,h \\right)

        :math:`D_1 / D_2` can be represented as 

        .. math:: sbdual \left ( e^{e\log{a}}, \\frac{\left ( af\log{a} + be \\right )
                        e^{elog(a)}}{a}, \\frac{\left ( ag\log{a} + ce \\right )
                        e^{elog(a)}}{a}, \\frac{(a^{h}\log(a)+a(bg + cf) + e(ad - bc)
                        + (af\log{a} + be)(ag\log{a} + ce))e^{e\log(a)}}{a^{2}}\\right ) .       
        """

        if type(other) in core.scalartypes:

            # Create a view to coeffs of x.
            coeffs = self.coeffs
            order = self.order                  # Get attributes.
            n = self.numcoeffs

            new = self.__class__(0)             # Create a new mdual number.

            lower = 1                           # Begin Taylor series with 1.

            # Allocate space for coefficients.
            v = np.zeros(n, np.complex128)         # Returning value.
            s = np.zeros(n, np.complex128)         # s = (z - x0)
            p = np.zeros(n, np.complex128)         # Powers of (z - x0). p = s**k.
            old = np.zeros(n, np.complex128)       # Space for copies.

            # Initialize variables and arrays.

            # Extract real part from "self".
            x0 = coeffs[0]

            s = coeffs.copy()
            s[0] = 0.
            p[0] = 1.
            v[0] = x0**other

            core.mdual_pow(v, s, p, old, x0, other, n, lower, order)

            new.coeffs = v                      # Overwrite attributes.
            new.order = order
            new.numcoeffs = n

        elif isinstance(other, type(self)):

            new = exp(other * log(self))

        return new

    def __rpow__(self, other):
        """
        :return: mdual

        To provide reverse power.

        It defines how to raise a real number to a dual power.

        :param self: multidual number
        :param other: real number
        :type self: mdual
        :type other: int, float

        .. code-block:: Python

            >>> a = mdual([1,2,3,4])
            >>> 2**a
            mdual([ 2., 2.77258872, 4.15888308,11.31061361])
        ..


        For a bidual number :math:`D`, where

        .. math::
           D = sdual\\left ( a,b,c,d \\right )
        
        and an arbitrary real number, :math:`r`, the general representation
        of a real number raised to a bidual power can be represented as
           
        .. math :: sdual \\left(e^{a\log{r}},
                                be^{a\log{r}}\log{r},
                                ce^{a\log{r}}\log{r},
                                (bc\log{r} + d)e^{a\log{r}}\log{r}
                                 \\right ).


        """

        new = exp(self * np.log(other))

        return new

    def get_cr(self, i, j):
        """
        Get value from the position [i,j] of the CR matrix form.

        This is the Cauchy Riemann binary mapping method for a
        multidual number. It can retrieve any position of the CR
        matrix.

        :param self: Cauchy-Riemann matrix
        :param i: First index of position
        :param j: Second index of position
        :type self: array-like
        :type i: int
        :type j: int

        .. code-block:: Python

            >>> t = mdual([1,8,2,7,3,6,4,5])
            >>> print(t.get_cr(3,4))
            -5

        ..
        """

        return core.mdual_get_cr(self.coeffs, i, j)

    def to_cr(self):
        """
        To convert a multidual number into its Cauchy-Riemann matrix form.

        :param self: A multidual value. Multidual values can
           be represented as a scalar, vector, or matrix.
        :type self: mdual
        :rtype: array-like

        .. code-block:: Python

            >>> a = mdual([1,2,3,4])
            >>> a.to_cr()
            marray([[ 1.,  0.,  0.,  0.],
                   [ 2.,  1.,  0.,  0.],
                   [ 3.,  0.,  1.,  0.],
                   [ 4.,  3.,  2.,  1.]])
        ..
        """

        n = self.numcoeffs                      # Get number of coefficients.
        coeffs = self.coeffs                    # Create a view to coeffs of x.
        array = np.zeros([n, n], np.complex128)    # Allocate space for CR matrix.

        core.mdual_to_cr(coeffs, n, array)      # Fill CR matrix.

        return array


class marray(core.marray):

    def __getitem__(self, index):
        """
        :return: int

        To get the value of a position in an array of multidual numbers.

        :param self: multidual number
        :param index: position of the desired value
        :type self: mdual
        :type index: int, array-like

        .. code-block:: Python

            >>> M = marray([[[1,2],[3,4]],
                            [[5,6],[7,8]]])
            >>> print(M[1,1])
            7 + 8*eps(1)
        
        ..
        """

        # Number of dimensions of the array.
        ndims = len(self.shape)

        # Case 1. Index is integer and array is 1D.
        if (type(index) in core.inttypes) and (ndims == 1):

            value = mdual(self.coeffs[index])

        # Case 2. Index is a tuple.
        elif isinstance(index, tuple):

            nidx = len(index)                   # Number of positions in index.

            if any(i is Ellipsis for i in index):

                raise IndexError("indices not valid for array")

            elif (nidx <= ndims + 1):

                all_int = all(type(i) in core.inttypes for i in index)

                if (nidx == ndims + 1) and all_int:

                    value = self.coeffs[index]

                elif (nidx == ndims) and all_int:

                    value = mdual(self.coeffs[index])

                elif (nidx <= ndims):

                    value = self.__getslice(index)

                else:

                    raise IndexError("indices not valid for array")

            else:

                raise IndexError("too many indices for array")

        # Case 3. Other cases.
        else:

            value = self.__getslice(index)

        return value

    def __mul__(self, other):
        """
        :return: marray

        This method provides multiplication of an array of multidual values with
        scalars and multidual numbers.

        :param self: multidual number
        :param other: scalar or multidual number to be multiplied with self
        :type self: marray
        :type other: mdual, int, float

        .. code-block:: Python

            >>> M = marray([[[1,2],[3,4]],
                            [[5,6],[7,8]]])
            >>> print(M * M)
            marray([[[ -3.,  4.],  [ -7., 24.]],
                    [[-11., 60.],  [-15.,112.]]])
        ..
        """

        new = self.__class__([[0]])             # New array for result.

        coeffs1 = self.coeffs                   # Get mdual attributes from
        order1 = self.order                     # self.
        n1 = self.numcoeffs

        shape1 = self.shape                     # Get self's array attributes.
        size1 = self.size                       # These won't change.

        if type(other) in core.scalartypes:

            # Case 1. Multiplication by scalar.

            new_coeffs = coeffs1.copy()

            new_coeffs *= other

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order1
            new.numcoeffs = n1
            new.shape = shape1
            new.size = size1

        elif isinstance(other, mdual):

            # Case 2. Multiplication by mdual.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            if (n1 >= n2):

                new_coeffs = np.zeros(shape1 + (n1,), np.complex128)
                order = order1
                n = n1

                for i in range(size1):

                    idx = core.item2index(i, shape1, size1)

                    core.mdual_mul(coeffs1[idx], n1,
                                   coeffs2, n2, new_coeffs[idx])

            else:

                new_coeffs = np.zeros(shape1 + (n2,), np.complex128)
                order = order2
                n = n2

                for i in range(size1):

                    idx = core.item2index(i, shape1, size1)

                    core.mdual_mul(
                        coeffs2, n2, coeffs1[idx], n1, new_coeffs[idx])

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order
            new.numcoeffs = n
            new.shape = shape1
            new.size = size1

        elif isinstance(other, type(self)):

            # Case 3. Element-wise multiplication

            # Get array attributes from other.
            shape2 = other.shape
            size2 = other.size

            if (shape1 != shape2):
                raise ValueError("operands could not be broadcast together " +
                                 "with shapes %s %s" % (shape1, shape2))

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            if (n1 >= n2):

                new_coeffs = np.zeros(shape1 + (n1,), np.complex128)
                order = order1
                n = n1

                for i in range(size1):

                    idx = core.item2index(i, shape1, size1)

                    core.mdual_mul(coeffs1[idx], n1,
                                   coeffs2[idx], n2, new_coeffs[idx])

            else:

                new_coeffs = np.zeros(shape1 + (n2,), np.complex128)
                order = order2
                n = n2

                for i in range(size1):

                    idx = core.item2index(i, shape1, size1)

                    core.mdual_mul(coeffs2[idx], n2,
                                   coeffs1[idx], n1, new_coeffs[idx])

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order
            new.numcoeffs = n
            new.shape = shape1
            new.size = size1

        else:                                   # Case 4. Incompatible types.

            raise TypeError("unsupported operand type(s) for +: '%s' and '%s'"
                            % (core.strtype(self), core.strtype(other)))

        return new

    def __truediv__(self, other):
        """
        :return: marray

        This method provides multiplication of an array of multidual values with
        scalars and multidual numbers.

        :param self: multidual number
        :param other: scalar or multidual number to be multiplied with self
        :type self: marray
        :type other: mdual, int, float

        .. code-block:: Python

            >>> M = marray([[[1,2],[3,4]],
                            [[5,6],[7,8]]])
            >>> N = marray([[9,10],[11,12]],
                           [[13,14],[15,16]])
            >>> print(M / N)
            marray([[[0.11111111,0.09876543],  [0.27272727,0.0661157 ]],
                    [[0.38461538,0.04733728],  [0.46666667,0.03555556]]])

        .. 
        """

        new = self.__class__([[0]])             # New array for result.

        coeffs1 = self.coeffs                   # Get mdual attributes from
        order1 = self.order                     # self.
        n1 = self.numcoeffs

        shape1 = self.shape                     # Get self's array attributes.
        size1 = self.size                       # These won't change.

        if type(other) in core.scalartypes:     # Case 1. Division by scalar.

            new_coeffs = coeffs1.copy()

            new_coeffs /= other

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order1
            new.numcoeffs = n1
            new.shape = shape1
            new.size = size1

        elif isinstance(other, mdual):

            # Case 2. Division by mdual.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            # Find max number of coefficients.
            if (n1 >= n2):
                order = order1
                n = n1
            else:
                order = order2
                n = n2

            # Allocate space for coefficients.
            num = np.zeros(shape1 + (n,), np.complex128)   # Numerator.
            den = np.zeros(n, np.complex128)               # Denominator.
            old = np.zeros(n, np.complex128)               # Space for copies.
            cden = np.zeros(n, np.complex128)              # Conjugated denominator.

            for i in range(size1):

                idx = core.item2index(i, shape1, size1)

                # Initialize numerator and denominator.
                num[idx + (slice(0, n1),)] = coeffs1[idx].copy()
                den[:n2] = coeffs2.copy()

                # Compute coefficients of the numerator.
                core.mdual_truediv(num[idx], den, old, cden, n, order2)

            new.coeffs = num                    # Overwrite attributes.
            new.order = order
            new.numcoeffs = n
            new.shape = shape1
            new.size = size1

        elif isinstance(other, type(self)):     # Case 3. Element-wise div.

            # Get array attributes from other.
            shape2 = other.shape
            size2 = other.size

            if (shape1 != shape2):
                raise ValueError("operands could not be broadcast together " +
                                 "with shapes %s %s" % (shape1, shape2))

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            # Find max number of coefficients.
            if (n1 >= n2):
                order = order1
                n = n1
            else:
                order = order2
                n = n2

            # Allocate space for coefficients.
            num = np.zeros(shape1 + (n,), np.complex128)   # Numerator.
            den = np.zeros(n, np.complex128)               # Denominator.
            old = np.zeros(n, np.complex128)               # Space for copies.
            cden = np.zeros(n, np.complex128)              # Conjugated denominator.

            for i in range(size1):

                idx = core.item2index(i, shape1, size1)

                # Initialize numerator and denominator.
                num[idx + (slice(0, n1),)] = coeffs1[idx].copy()
                den[:n2] = coeffs2[idx].copy()

                # Compute coefficients of the numerator.
                core.mdual_truediv(num[idx], den, old, cden, n, order)

            new.coeffs = num                    # Overwrite attributes.
            new.order = order
            new.numcoeffs = n
            new.shape = shape1
            new.size = size1

        else:                                   # Case 4. Incompatible types.

            raise TypeError("unsupported operand type(s) for +: '%s' and '%s'"
                            % (core.strtype(self), core.strtype(other)))

        return new

    def __rtruediv__(self, other):
        """
        :return: marray

        To provide reverse division for multidual arrays.

        It defines how to make a division between a scalar and a
        multidual array. This function creates a new array
        whose result is the division of the scalar by each
        element of the original array.

        :type self: int, float ???
        :type other: marray

        .. code-block:: Python

            >>> M = marray([[[1,2],[3,4]],
                            [[5,6],[7,8]]])
            >>> print(2 / M)
            marray([[[ 2.        ,-4.        ],
                     [ 0.66666667,-0.88888889]],

                    [[ 0.4       ,-0.48      ],
                     [ 0.28571429,-0.32653061]]])
        ..
        """

        new = self.__class__([[0]])             # New array for result.

        coeffs = self.coeffs                    # Create a view to self.coeffs.
        order = self.order                      # Get attributes.
        n = self.numcoeffs

        shape = self.shape                      # Get self's array attributes.
        size = self.size                        # These won't change.

        # Allocate space for coefficients.
        num = np.zeros(shape + (n,), np.complex128)    # Numerator.
        den = np.zeros(n, np.complex128)               # Denominator.
        old = np.zeros(n, np.complex128)               # Space for copies.
        cden = np.zeros(n, np.complex128)              # Conjugated denominator.

        for i in range(size):

            idx = core.item2index(i, shape, size)

            # Initialize numerator and denominator.
            num[idx + (0,)] = other
            den = coeffs[idx].copy()

            # Compute coefficients of the numerator.
            core.mdual_truediv(num[idx], den, old, cden, n, order)

        new.coeffs = num                        # Overwrite attributes.
        new.order = order
        new.numcoeffs = n

        return new

    def __pow__(self, e):
        """
        :return: marray

        Provides the power function method for multidual arrays.

        This function makes use of the Taylor series expansion approach, which is
        exact for multidual numbers.

        :param self: multidual array
        :param e: multidual array to raise the multidual array by
        :type self: marray
        :type e: marray

        .. code-block:: Python

            >>> u = marray([[  1,  2,  3],
                            [  4,  5,  6],
                            [  7,  8,  9],
                            [ 10, 11, 12],
                            [ 13, 14, 15]])
            >>> print(u**2)
            marray([[  1.,  4.,  6., 12.],
                    [ 16., 40., 48., 60.],
                    [ 49.,112.,126.,144.],
                    [100.,220.,240.,264.],
                    [169.,364.,390.,420.]])
        ..
        """

        new = self.__class__([[0]])             # New array for result.

        coeffs = self.coeffs                    # Get mdual attributes from
        order = self.order                      # self.
        n = self.numcoeffs

        shape = self.shape                      # Get self's array attributes.
        size = self.size

        lower = 1                               # Begin Taylor series with 1.

        # Allocate space for coefficients.

        # Coefficients of the new array.
        new_coeffs = np.zeros(shape + (n,), np.complex128)

        s = np.empty(n, np.complex128)             # s = (z - x0)
        p = np.empty(n, np.complex128)             # Powers of (z - x0). p = s**k.
        old = np.empty(n, np.complex128)           # Space for copies.

        for i in range(size):

            idx = core.item2index(i, shape, size)

            # Initialize variables and arrays.

            x0 = coeffs[idx + (0,)]             # Extract real part.

            s[:n] = coeffs[idx, :].copy()

            s[0] = 0.

            p[0] = 1.
            p[1:] = 0.

            v = new_coeffs[idx]
            v[0] = x0**e

            core.mdual_pow(v, s, p, old, x0, e, n, lower, order)

        new.coeffs = new_coeffs                 # Overwrite attributes.
        new.order = order
        new.numcoeffs = n
        new.shape = shape
        new.size = size

        return new

    def to_cr(self):
        """
        To convert a multidual into its Cauchy-Riemann matrix form.
        
        :param self: A multidual. multidual values can
           be represented as a scalar, vector, or matrix.
        
        :type self: mdual
        :rtype: array-like

        .. code-block :: Python
            >>> u = marray([[  1,  2,  3],
                            [  4,  5,  6]])
            >>> print(u.to_cr())
            [1. 4. 2. 5. 3. 6. 0. 0.]
        ..
        """

        coeffs = self.coeffs                    # Create a view to coeffs.
        numcoeffs = self.numcoeffs              # Get array attributes.
        shape = self.shape
        ndims = len(shape)                      # Number of array's dimensions.
        size0 = shape[0]                        # Size of the 1st dimension.

        n = numcoeffs * size0                   # Side of the new array.

        if (ndims == 1):

            new = np.zeros(n)                   # Allocate space.

            for k in range(numcoeffs):          # For each mdual coefficient
                for p in range(size0):          # For each vector position.

                    new[size0 * k + p] = coeffs[p, k]

        elif (ndims == 2) and (shape[0] == shape[1]):

            new = np.zeros((n, n))              # Allocate space.

            for i in range(numcoeffs):          # For each Cauchy-Riemann row.
                for j in range(i + 1):          # For each CR column.
                    for p in range(size0):      # For each matrix row.
                        for q in range(size0):  # For each matrix column.

                            new[size0 * i + p, size0 * j + q] = \
                                core.mdual_get_cr(coeffs[p, q], i, j)

                            # Note: the CR upper triangular
                            # of a mdual is zero.

        else:

            raise IndexError(f"invalid shape {shape}. " +
                             "Array must be 2D square or 1D.")

        return new


def eps(imagDirs, order=0):
    """
    To create a multidual number with value 1 at the specified imaginary direction.

    :param imagDirs: The imaginary direction of value 1. This location given can
       be given as a scalar or array, depending on the desired location.
    :param order: Order of the multidual number
    :type imagDirs: int or array-like
    :type order: int, optional, default=0.

    .. code-block :: Python

        >>> eps(2)
        mdual([0, 0, 1, 0])
        >>> eps([1,2])
        mdual([0, 0, 0, 1])
    ..    
    """

    new = mdual(0)                              # New mdual number.

    pos = getPos(imagDirs)                      # Get the coefficient position.

    # If order is not defined, compute it.
    if (order == 0):
        order = (pos).bit_length()              # ( = int(ceil(log2(pos+1))) )

    n = 1 << order                              # Compute num. of coefficients.

    new_coeffs = np.zeros(n, np.complex128)        # Allocate space for coeffs.

    new_coeffs[pos] = 1.                        # Assign unary value.

    new.coeffs = new_coeffs                     # Overwrite attributes
    new.order = order
    new.numcoeffs = n

    return new


def mzero(order=0):
    """
    To create a multidual number of the given order whose coefficients 
        are all zeros.

    :param order: The order of the multidual number
    :type order: int, optional, default=0

    .. code-block:: Python

       >>> a = mzero(1)
       >>> b = mzero(2)
       >>> a
       0 + 0*eps(1)
       >>> b
       0 + 0*eps(1) + 0*eps(2) + 0*eps([1,2])
    ..
    """

    new = mdual(0)                              # Create a new mdual number.
    n = 1 << order                              # Number of coeffs.
    coeffs = np.zeros(n, np.complex128)            # Allocate space.

    new.coeffs = coeffs                         # Overwrite information.
    new.order = order
    new.numcoeffs = n

    return new


def zeros(shape, order=0):
    """
    Create an array filled with null multidual numbers of the given 
        order.
 
    :param shape: Shape of the container of zeros
    :param order: Order of the container of zeros
    :type shape: int
    :type order: int, optional, default=0

    .. code-block:: Python

        >>> a = zeros(1,1)
        >>> a
        marray([[0.,0.]])
        >>> b = zeros(2,1)
        >>> b
        marray([[0.,0.]
                [0.,0.]])
    ..
    """

    new = marray([[0]])                         # Create a new mdual array.
    n = 1 << order                              # Number of coeffs per entry.

    # Compute array attributes.

    if isinstance(shape, int):                  # Case 1: array 1D.
        new_size = shape
        new_shape = tuple([shape])
        np_shape = list([shape, n])
    else:                                       # Case 2: array 2D.
        new_size = np.prod(shape)
        np_shape = list(shape) + list([n])
        new_shape = tuple(shape)

    coeffs = np.zeros(np_shape, np.complex128)     # Create coeffs array.

    new.coeffs = coeffs                         # Overwrite information.
    new.order = order
    new.numcoeffs = n
    new.shape = new_shape
    new.size = new_size

    return new


def rarray(coef_list, order=0):
    """
    To create an array of multiduals from an array of real
    numbers.
      
    The constructor of the array class is not designed to
    parse arrays of real numbers as multidualss. Use this
    function to make the conversion.

    :param coef_list: A list of coefficients to be applied to an array of
       multidualss
    :param order: Order of the multidual array
    :type coef_list: array-like
    :type order: int, optional, default=0

    .. code-block :: Python

        >>> a = rarray([1,0],0)
        >>> a
        marray([[1.],
                [0.]])
        >>> b = rarray([1,0],1)
        >>> b
        marray([[1.,0.],
                [0.,0.]])
    ..
    """

    new = marray([[0]])                         # Create a new mdual array.
    n = 1 << order                              # Number of coeffs per entry.

    shape = core.getshape(coef_list)            # Get array attributes.
    size = np.prod(shape)

    ndims = len(shape)                          # Get its number of dimensions.

    # Allocate space.
    coeffs = np.zeros(shape + (n,), np.complex128)

    for i in range(size):

        idx = core.item2index(i, shape, size)   # Get index of coef_list.
        idx0 = idx + (0,)                       # Set index in coeffs.

        # Initialize value. View to coef_list.
        value = coef_list[idx[0]]

        for j in range(1, ndims):
            value = value[idx[j]]               # Reduce window to coef_list.

        coeffs[idx0] = value

    new.coeffs = coeffs                         # Overwrite attributes.
    new.order = order
    new.numcoeffs = n
    new.shape = shape
    new.size = size

    return new


def exp(x):
    """
    Provide exponential function for multidual numbers.
    
    This function is based on the Taylor series expansion
    approach, which is exact for multidual numbers.
    
    :param x: Input of the exponential function
    :type x: mdual

    .. code-block :: Python
    
    >>> m = mdual([1,2,3,4])
    >>> u = exp(m)
    >>> print(u)
    2.71828 + 5.43656*eps(1) + 8.15485*eps(2) + 27.1828*eps([1, 2])
    ..
    """

    if type(x) in core.scalartypes:

        new = mdual(0)                          # Create a new mdual number.

        new.coeffs[0] = np.exp(x)

    elif isinstance(x, mdual):

        coeffs = x.coeffs                       # Create a view to coeffs of x.
        order = x.order                         # Get attributes.
        n = x.numcoeffs

        lower = 1                               # Begin Taylor series with 1.

        new = mdual(0)                          # Create a new mdual number.

        # Allocate space for coefficients.
        v = np.zeros(n, np.complex128)             # Returning value.
        s = np.zeros(n, np.complex128)             # s = (z - x0)
        p = np.zeros(n, np.complex128)             # Powers of (z - x0). p = s**k.
        old = np.zeros(n, np.complex128)           # Space for copies.

        # Initialize variables and arrays.

        x0 = coeffs[0]                          # Extract self's real part.

        s = coeffs.copy()
        s[0] = 0.
        p[0] = 1.
        v[0] = 1.

        core.mdual_exp(v, s, p, old, x0, n, lower, order)

        new.coeffs = v                          # Overwrite attributes.
        new.order = order
        new.numcoeffs = n

    elif isinstance(x, marray):

        coeffs = x.coeffs                       # Create a view to coeffs of x.
        order = x.order                         # Get attributes.
        n = x.numcoeffs
        shape = x.shape
        size = x.size

        lower = 1                               # Begin Taylor series with 1.

        new = marray([[0]])                     # Create a new mdual array.

        # Allocate space for coefficients.

        # Coefficients of the new array.
        new_coeffs = np.zeros(shape + (n,), np.complex128)

        s = np.empty(n, np.complex128)             # s = (z - x0)
        p = np.empty(n, np.complex128)             # Powers of (z - x0). p = s**k.
        old = np.empty(n, np.complex128)           # Space for copies.

        for i in range(size):

            idx = core.item2index(i, shape, size)

            # Initialize variables and arrays.

            x0 = coeffs[idx + (0,)]             # Extract real part.

            s[:n] = coeffs[idx, :].copy()

            s[0] = 0.

            p[0] = 1.
            p[1:] = 0.

            v = new_coeffs[idx]                 # Coeffs of a single position.
            v[0] = 1.

            core.mdual_exp(v, s, p, old, x0, n, lower, order)

        new.coeffs = new_coeffs                 # Overwrite attributes.
        new.order = order
        new.numcoeffs = n
        new.shape = shape
        new.size = size

    return new


def log(x):
    """
    Provide logarithm function for multidual numbers.
    
    Wrapper for the logarithm of the Cauchy Riemann matrix
    version of a multidual number.
    
    :param x: Input of the logarithm function
    :type x: mdual

    .. code-block :: Python
      
        >>> m = mdual([1,2,3,4])
        >>> u = log(m)
        >>> print(u)   
        0 + 2*eps(1) + 3*eps(2) - 2*eps([1, 2])
    ..
    """

    if type(x) in core.scalartypes:

        new = mdual(0)                          # Create a new mdual number.

        new.coeffs[0] = np.log(x)

    elif isinstance(x, mdual):

        coeffs = x.coeffs                       # Create a view to coeffs of x.
        order = x.order                         # Get attributes.
        n = x.numcoeffs

        lower = 1                               # Begin Taylor series with 1.

        new = mdual(0)                          # Create a new mdual number.

        # Allocate space for coefficients.
        v = np.zeros(n, np.complex128)             # Returning value.
        s = np.zeros(n, np.complex128)             # s = (z - x0)
        p = np.zeros(n, np.complex128)             # Powers of (z - x0). p = s**k.
        old = np.zeros(n, np.complex128)           # Space for copies.

        # Initialize variables and arrays.

        x0 = coeffs[0]                          # Extract self's real part.

        s = coeffs.copy()
        s[0] = 0.
        p[0] = 1.
        v[0] = np.log(x0)

        core.mdual_log(v, s, p, old, x0, n, lower, order)

        new.coeffs = v                          # Overwrite attributes.
        new.order = order
        new.numcoeffs = n

    elif isinstance(x, marray):

        coeffs = x.coeffs                       # Create a view to coeffs of x.
        order = x.order                         # Get attributes.
        n = x.numcoeffs
        shape = x.shape
        size = x.size

        lower = 1                               # Begin Taylor series with 1.

        new = marray([[0]])                     # Create a new mdual array.

        # Allocate space for coefficients.

        # Coefficients of the new array.
        new_coeffs = np.zeros(shape + (n,), np.complex128)

        s = np.empty(n, np.complex128)             # s = (z - x0)
        p = np.empty(n, np.complex128)             # Powers of (z - x0). p = s**k.
        old = np.empty(n, np.complex128)           # Space for copies.

        for i in range(size):

            idx = core.item2index(i, shape, size)

            # Initialize variables and arrays.

            x0 = coeffs[idx + (0,)]             # Extract real part.

            s[:n] = coeffs[idx, :].copy()

            s[0] = 0.

            p[0] = 1.
            p[1:] = 0.

            v = new_coeffs[idx]                 # Coeffs of a single position.
            v[0] = np.log(x0)

            core.mdual_log(v, s, p, old, x0, n, lower, order)

        new.coeffs = new_coeffs                 # Overwrite attributes.
        new.order = order
        new.numcoeffs = n
        new.shape = shape
        new.size = size

    # End if. Variable type cases.

    return new


def sin(x):
    """
    Provide sine function for multidual numbers.
    
    This function is based on the Taylor series expansion
    approach, which is exact for multidual numbers.
    
    :param x: Input of the sine function
    :type x: mdual  
    
    .. code-block :: Python

        >>> m = mdual([1,2,3,4])
        >>> u = sin(m)
        >>> print(u)
        0.841471 + 1.0806*eps(1) + 1.62091*eps(2) - 2.88762*eps([1, 2])
    ..
   """

    if type(x) in core.scalartypes:

        new = mdual(0)                          # Create a new mdual number.

        new.coeffs[0] = np.sin(x)

    elif isinstance(x, mdual):

        coeffs = x.coeffs                       # Create a view to coeffs of x.
        order = x.order                         # Get attributes.
        n = x.numcoeffs

        lower = 1                               # Begin Taylor series with 1.

        new = mdual(0)                          # Create a new mdual number.

        # Allocate space for coefficients.
        v = np.zeros(n, np.complex128)             # Returning value.
        s = np.zeros(n, np.complex128)             # s = (z - x0)
        p = np.zeros(n, np.complex128)             # Powers of (z - x0). p = s**k.
        old = np.zeros(n, np.complex128)           # Space for copies.

        # Initialize variables and arrays.

        x0 = coeffs[0]                          # Extract self's real part.

        s = coeffs.copy()
        s[0] = 0.
        p[0] = 1.
        v[0] = np.sin(x0)

        core.mdual_sin(v, s, p, old, x0, n, lower, order)

        new.coeffs = v                          # Overwrite attributes.
        new.order = order
        new.numcoeffs = n

    elif isinstance(x, marray):

        coeffs = x.coeffs                       # Create a view to coeffs of x.
        order = x.order                         # Get attributes.
        n = x.numcoeffs
        shape = x.shape
        size = x.size

        lower = 1                               # Begin Taylor series with 1.

        new = marray([[0]])                     # Create a new mdual array.

        # Allocate space for coefficients.

        # Coefficients of the new array.
        new_coeffs = np.zeros(shape + (n,), np.complex128)

        s = np.empty(n, np.complex128)             # s = (z - x0)
        p = np.empty(n, np.complex128)             # Powers of (z - x0). p = s**k.
        old = np.empty(n, np.complex128)           # Space for copies.

        for i in range(size):

            idx = core.item2index(i, shape, size)

            # Initialize variables and arrays.

            x0 = coeffs[idx + (0,)]             # Extract real part.

            s[:n] = coeffs[idx, :].copy()

            s[0] = 0.

            p[0] = 1.
            p[1:] = 0.

            v = new_coeffs[idx]
            v[0] = np.sin(x0)

            core.mdual_sin(v, s, p, old, x0, n, lower, order)

        new.coeffs = new_coeffs                 # Overwrite attributes.
        new.order = order
        new.numcoeffs = n
        new.shape = shape
        new.size = size

    # End if. Variable type cases.

    return new


def cos(x):
    """
    Provide cosine function for multidual numbers.
   
    This function is based on the Taylor series expansion
    approach, which is exact for multidual numbers.
   
    :param x: Input of the cosine function
    :type x: mdual  

    .. code-block :: Python

        >>> m = mdual([1,2,3,4])
        >>> u = cos(m)
        >>> print(u)
        0.540302 - 1.68294*eps(1) - 2.52441*eps(2) - 6.6077*eps([1, 2])
    ..
    """

    if type(x) in core.scalartypes:

        new = mdual(0)                          # Create a new mdual number.

        new.coeffs[0] = np.cos(x)

    elif isinstance(x, mdual):

        coeffs = x.coeffs                       # Create a view to coeffs of x.
        order = x.order                         # Get attributes.
        n = x.numcoeffs

        lower = 1                               # Begin Taylor series with 1.

        new = mdual(0)                          # Create a new mdual number.

        # Allocate space for coefficients.
        v = np.zeros(n, np.complex128)             # Returning value.
        s = np.zeros(n, np.complex128)             # s = (z - x0)
        p = np.zeros(n, np.complex128)             # Powers of (z - x0). p = s**k.
        old = np.zeros(n, np.complex128)           # Space for copies.

        # Initialize variables and arrays.

        x0 = coeffs[0]                          # Extract self's real part.

        s = coeffs.copy()
        s[0] = 0.
        p[0] = 1.
        v[0] = np.cos(x0)

        core.mdual_cos(v, s, p, old, x0, n, lower, order)

        new.coeffs = v                          # Overwrite attributes.
        new.order = order
        new.numcoeffs = n

    elif isinstance(x, marray):

        coeffs = x.coeffs                       # Create a view to coeffs of x.
        order = x.order                         # Get attributes.
        n = x.numcoeffs
        shape = x.shape
        size = x.size

        lower = 1                               # Begin Taylor series with 1.

        new = marray([[0]])                     # Create a new mdual array.

        # Allocate space for coefficients.

        # Coefficients of the new array.
        new_coeffs = np.zeros(shape + (n,), np.complex128)

        s = np.empty(n, np.complex128)             # s = (z - x0)
        p = np.empty(n, np.complex128)             # Powers of (z - x0). p = s**k.
        old = np.empty(n, np.complex128)           # Space for copies.

        for i in range(size):

            idx = core.item2index(i, shape, size)

            # Initialize variables and arrays.

            x0 = coeffs[idx + (0,)]             # Extract real part.

            s[:n] = coeffs[idx, :].copy()

            s[0] = 0.

            p[0] = 1.
            p[1:] = 0.

            v = new_coeffs[idx]
            v[0] = np.cos(x0)

            core.mdual_cos(v, s, p, old, x0, n, lower, order)

        new.coeffs = new_coeffs                 # Overwrite attributes.
        new.order = order
        new.numcoeffs = n
        new.shape = shape
        new.size = size

    return new


def sqrt(x):
    """
    Provide sqrt function for multidual numbers and arrays.
    
    This function is based on the Taylor series expansion
    approach, which is exact for multidual numbers.
    
    :param x: Input of the square root function
    :type x: mdual  
    
    .. code-block :: Python

        >>> m = mdual([1,2,3,4])
        >>> u = sqrt(m)
        >>> print(u)
        1 + 1*eps(1) + 1.5*eps(2) + 0.5*eps([1, 2])
    ..
    """
    if type(x) in core.scalartypes:

        new = mdual(0)                          # Create a new mdual number.

        new.coeffs[0] = np.sqrt(x)

    elif isinstance(x, mdual):

        coeffs = x.coeffs                       # Create a view to coeffs of x.
        order = x.order                         # Get attributes.
        n = x.numcoeffs

        e0 = 0.5                                # Exponent for square root.
        lower = 1                               # Begin Taylor series with 1.

        new = mdual(0)                          # Create a new mdual number.

        # Allocate space for coefficients.
        v = np.zeros(n, np.complex128)             # Returning value.
        s = np.zeros(n, np.complex128)             # s = (z - x0)
        p = np.zeros(n, np.complex128)             # Powers of (z - x0). p = s**k.
        old = np.zeros(n, np.complex128)           # Space for copies.

        # Initialize variables and arrays.

        x0 = coeffs[0]                          # Extract self's real part.

        s = coeffs.copy()
        s[0] = 0.
        p[0] = 1.
        v[0] = x0**e0

        core.mdual_pow(v, s, p, old, x0, e0, n, lower, order)

        new.coeffs = v                          # Overwrite attributes.
        new.order = order
        new.numcoeffs = n

    elif isinstance(x, marray):

        coeffs = x.coeffs                       # Create a view to coeffs of x.
        order = x.order                         # Get attributes.
        n = x.numcoeffs
        shape = x.shape
        size = x.size

        e0 = 0.5                                # Exponent for square root.
        lower = 1                               # Begin Taylor series with 1.

        new = marray([[0]])                     # Create a new mdual array.

        # Allocate space for coefficients.

        # Coefficients of the new array.
        new_coeffs = np.zeros(shape + (n,), np.complex128)

        s = np.empty(n, np.complex128)             # s = (z - x0)
        p = np.empty(n, np.complex128)             # Powers of (z - x0). p = s**k.
        old = np.empty(n, np.complex128)           # Space for copies.

        for i in range(size):

            idx = core.item2index(i, shape, size)

            # Initialize variables and arrays.

            x0 = coeffs[idx + (0,)]             # Extract real part.

            s[:n] = coeffs[idx, :].copy()

            s[0] = 0.

            p[0] = 1.
            p[1:] = 0.

            v = new_coeffs[idx]
            v[0] = x0**e0

            core.mdual_pow(v, s, p, old, x0, e0, n, lower, order)

        new.coeffs = new_coeffs                 # Overwrite attributes.
        new.order = order
        new.numcoeffs = n
        new.shape = shape
        new.size = size

    return new


def dot(array1, array2):
    """
    Dot product for n-dimensional arrays of multidual numbers.
 
    :param array1: n-dimensional array of multidual numbers
    :param array2: n-dimensional array of multidual numbers
    :type array1: mdual  
    :type array2: mdual  
    
    .. code-block:: Python
    
       >>> u = marray([[1,2],
                       [3,4]])
       >>> v = marray([[5,6],
                       [7,8]])
       >>> M = marray([[[1,2],[3,4]],
                       [[5,6],[7,8]]])
       >>> N = marray([[[ 9,10],[11,12]],
                       [[13,14],[15,16]]])
       
       >>> print(dot(u, v))
       26 + 68*eps(1)
       >>> print(dot(M, u))
       marray([[10.,28.],
               [26.,68.]])
       >>> print(dot(u, M))
       marray([[16.,42.],
               [24.,62.]])
       >>> print(dot(M, N))
       marray([[[ 48.,122.],  [ 56.,142.]],
               [[136.,306.],  [160.,358.]]])
 
    """

    # Shapes alignment verification.

    shape1 = array1.shape                       # Get shapes.
    shape2 = array2.shape
    size1 = array1.size                         # Get sizes.
    size2 = array2.size
    ndims1 = len(shape1)                        # Get number of dimensions.
    ndims2 = len(shape2)
    dim1 = max(ndims1 - 1, 0)                   # Pos. last dim. array1.
    dim2 = max(ndims2 - 2, 0)                   # Pos. 2nd to last dim. array2.
    n1 = shape1[dim1]                           # Size of dim1.
    n2 = shape2[dim2]                           # Size of dim2.

    if (n1 != n2):                              # Size verification.
        raise ValueError(
            "shapes %s and %s not aligned: %d (dim %d) != %d (dim %d)"
            % (shape1, shape2, n1, dim1, n2, dim2))

    n = n1                                      # Size collapsing dimension.

    coeffs1 = array1.coeffs                     # View to coeffs.
    coeffs2 = array2.coeffs
    order1 = array1.order                       # Order
    order2 = array2.order
    nc1 = array1.numcoeffs                      # Number of coefficients.
    nc2 = array2.numcoeffs

    # Get shape of the new array.
    new_shape = shape1[:dim1] + shape2[:dim2] + shape2[dim2 + 1:]

    if not new_shape:

        # Case 1. Dot product between two vectors.

        new = mdual(0)                          # Create a new mdual number.

        if (nc1 >= nc2):                        # Case 1.1.

            order = order1                      # Set attributes for result.
            nc = nc1

            temp = np.zeros((n, nc), np.complex128)

            core.mduarray_mul(coeffs1, nc1, coeffs2, nc2, temp)

            new_coeffs = sum(temp)

        else:                                   # Case 1.2.

            order = order2                      # Set attributes for result.
            nc = nc2

            temp = np.zeros((n, nc), np.complex128)

            core.mduarray_mul(coeffs2, nc2, coeffs1, nc1, temp)

            new_coeffs = sum(temp)

        new.coeffs = new_coeffs                 # Overwrite information.
        new.order = order
        new.numcoeffs = nc

    else:

        # Case 2. Mat-vec, vec-mat, mat-mat.

        new = marray([[0]])                     # Create a new mdual array.

        new_size = np.prod(new_shape)           # Size of the result.

        # Get size of the last dimension of array2.
        if (ndims2 == 1):
            p = 1                               # array2 is vector.
        else:
            p = shape2[ndims2 - 1]              # array2 is matrix.

        # Get partial sizes.
        psize1 = size1 // n                     # Num. "rows" of n elements.
        psize2 = size2 // n                     # Num. "columns" of n elements.

        if (nc1 >= nc2):                        # Case 2.1

            order = order1                      # Set attributes for result.
            nc = nc1

            # Allocate space.
            new_coeffs = np.zeros(new_shape + (nc,), np.complex128)
            temp = np.zeros(nc, np.complex128)

            for i in range(psize1):
                for j in range(psize2):

                    idx3 = core.item2index(i * p + j, new_shape, new_size)

                    for k in range(n):

                        idx1 = core.item2index(i * n + k, shape1, size1)
                        idx2 = core.item2index(j + k * p, shape2, size2)

                        core.mdual_mul_rs(
                            coeffs1[idx1], nc1, coeffs2[idx2], nc2, temp)

                        new_coeffs[idx3] += temp

        else:                                   # Case 2.2.

            order = order2                      # Set attributes for result.
            nc = nc2

            # Allocate space.
            new_coeffs = np.zeros(new_shape + (nc,), np.complex128)
            temp = np.zeros(nc, np.complex128)

            for i in range(psize1):
                for j in range(psize2):

                    idx3 = core.item2index(i * p + j, new_shape, new_size)

                    for k in range(n):

                        idx1 = core.item2index(i * n + k, shape1, size1)
                        idx2 = core.item2index(j + k * p, shape2, size2)

                        core.mdual_mul_rs(
                            coeffs2[idx2], nc2, coeffs1[idx1], nc1, temp)

                        new_coeffs[idx3] += temp

        new.coeffs = new_coeffs                 # Overwrite information.
        new.order = order
        new.numcoeffs = nc
        new.shape = new_shape
        new.size = new_size

    return new


def old_dot(array1, array2):
    """
    Dot product for n-dimensional arrays of multidual numbers.

    :param array1: n-dimensional array of multidual numbers
    :param array2: n-dimensional array of multidual numbers
    :type array1: mdual  
    :type array2: mdual  
   """

    # Shapes alignment verification.

    shape1 = array1.shape                       # Get shapes.
    shape2 = array2.shape
    size1 = array1.size                         # Get sizes.
    size2 = array2.size
    ndims1 = len(shape1)                        # Get number of dimensions.
    ndims2 = len(shape2)
    dim1 = max(ndims1 - 1, 0)                   # Pos. last dim. array1.
    dim2 = max(ndims2 - 2, 0)                   # Pos. 2nd to last dim. array2.
    n1 = shape1[dim1]                           # Size of dim1.
    n2 = shape2[dim2]                           # Size of dim2.

    if (n1 != n2):                              # Size verification.
        raise ValueError(
            "shapes %s and %s not aligned: %d (dim %d) != %d (dim %d)"
            % (shape1, shape2, n1, dim1, n2, dim2))

    n = n1                                      # Size collapsing dimension.

    coeffs1 = array1.coeffs                     # View to coeffs.
    coeffs2 = array2.coeffs
    order1 = array1.order                       # Order
    order2 = array2.order
    nc1 = array1.numcoeffs                      # Number of coefficients.
    nc2 = array2.numcoeffs

    # Get shape of the new array.
    new_shape = shape1[:dim1] + shape2[:dim2] + shape2[dim2 + 1:]

    if not new_shape:

        # Case 1. Dot product between two vectors.

        new = mdual(0)                          # Create a new mdual number.

        if (nc1 >= nc2):                        # Case 1.1.

            order = order1                      # Set attributes for result.
            nc = nc1
            new_coeffs = np.zeros(nc, np.complex128)
            temp = np.zeros(nc, np.complex128)

            for i in range(n):
                core.mdual_mul_rs(coeffs1[i], nc1, coeffs2[i], nc2, temp)
                new_coeffs += temp

        else:                                   # Case 1.2.

            order = order2                      # Set attributes for result.
            nc = nc2
            new_coeffs = np.zeros(nc, np.complex128)
            temp = np.zeros(nc, np.complex128)

            for i in range(n):
                core.mdual_mul_rs(coeffs2[i], nc2, coeffs1[i], nc1, temp)
                new_coeffs += temp

        new.coeffs = new_coeffs                 # Overwrite information.
        new.order = order
        new.numcoeffs = nc

    else:

        # Case 2. Mat-vec, vec-mat, mat-mat.

        new = marray([[0]])                     # Create a new mdual array.

        new_size = np.prod(new_shape)           # Size of the result.

        # Get size of the last dimension of array2.
        if (ndims2 == 1):
            p = 1                               # array2 is vector.
        else:
            p = shape2[ndims2 - 1]              # array2 is matrix.

        # Get partial sizes.
        psize1 = size1 // n                     # Num. "rows" of n elements.
        psize2 = size2 // n                     # Num. "columns" of n elements.

        if (nc1 >= nc2):                        # Case 2.1

            order = order1                      # Set attributes for result.
            nc = nc1

            # Allocate space.
            new_coeffs = np.zeros(new_shape + (nc,), np.complex128)
            temp = np.zeros(nc, np.complex128)

            for i in range(psize1):
                for j in range(psize2):

                    idx3 = core.item2index(i * p + j, new_shape, new_size)

                    for k in range(n):

                        idx1 = core.item2index(i * n + k, shape1, size1)
                        idx2 = core.item2index(j + k * p, shape2, size2)

                        core.mdual_mul_rs(
                            coeffs1[idx1], nc1, coeffs2[idx2], nc2, temp)

                        new_coeffs[idx3] += temp

        else:                                   # Case 2.2.

            order = order2                      # Set attributes for result.
            nc = nc2

            # Allocate space.
            new_coeffs = np.zeros(new_shape + (nc,), np.complex128)
            temp = np.zeros(nc, np.complex128)

            for i in range(psize1):
                for j in range(psize2):

                    idx3 = core.item2index(i * p + j, new_shape, new_size)

                    for k in range(n):

                        idx1 = core.item2index(i * n + k, shape1, size1)
                        idx2 = core.item2index(j + k * p, shape2, size2)

                        core.mdual_mul_rs(
                            coeffs2[idx2], nc2, coeffs1[idx1], nc1, temp)

                        new_coeffs[idx3] += temp

        new.coeffs = new_coeffs                 # Overwrite information.
        new.order = order
        new.numcoeffs = nc
        new.shape = new_shape
        new.size = new_size

    return new


def mcr_to_mnumber(x):
    """
    To convert a Cauchy-Riemann matrix to multidual number.

    :param x: Cauchy-Riemann matrix
    :type x: array-like

    .. code-block :: Python

        >>> M = mdual([1,2,3,4])
        >>> m_cr = M.to_cr()
        >>> m = mcr_to_mnumber(m_cr)
        >>> print(m)
        1 + 2*eps(1) + 3*eps(2) + 4*eps([1, 2])
    ..
    """

    new = mdual(x[:, 0])

    return new


def mcr_to_marray(x, order):
    """
    To convert a Cauchy-Riemann expanded vector or square
    matrix to array of mdual.

    :param x: Cauchy-Riemann expanded vector, or square matrix
    :param order: order of square matrix
    :type x: array-like
    :type order: int
    :rtype: mdual array

    .. code-block :: Python 

        >>> M = marray([[[1,2],[3,4]],
                        [[5,6],[7,8]]])
        >>> M_full = M.to_cr()
        >>> print(mcr_to_marray(M_full, order=1))
        marray([[[1.,2.],  [3.,4.]],
                [[5.,6.],  [7.,8.]]])
    ..
    """

    new = marray([[0]])

    numcoeffs = 1 << order                      # Number of mcomplex coeffs.

    np_shape = x.shape                          # Shape of the input array.
    ndims = len(np_shape)                       # Num. dimensions of the array.

    # Size (mdual) of the 1st dimension.
    size0 = np_shape[0] // numcoeffs

    if (ndims == 1):

        new_coeffs = np.zeros((size0, numcoeffs))

        shape = (size0,)
        size = size0

        for k in range(numcoeffs):              # For each mcomplex coefficient
            for p in range(size0):              # For each vector position.

                new_coeffs[p, k] = x[size0 * k + p]

    elif (ndims == 2) and (np_shape[0] == np_shape[1]):

        new_coeffs = np.zeros((size0, size0, numcoeffs))

        shape = (size0, size0)
        size = size0 * size0

        for k in range(numcoeffs):              # For each mdual coefficient
            for p in range(size0):              # For each matrix row.
                for q in range(size0):          # For each matrix column.

                    new_coeffs[p, q, k] = x[size0 * k + p, q]

    else:

        raise IndexError(
            "invalid shape %s. Array must be 2D square or 1D." % str(np_shape))

    new.coeffs = new_coeffs                     # Overwrite attributes.
    new.order = order
    new.numcoeffs = numcoeffs
    new.shape = shape
    new.size = size

    return new
