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
Module name:  mcomplex.

Description:  This module defines the subclasses and exclusive code of the
              multicomplex algebra.

Date:         May 5, 2016
Updated:      March 16, 2021
'''

import numpy as np

import multiZ.core as core

from multiZ.core import getImagDirs, getPos


class mcomplex(core.mnumber):

    def __str__(self):
        """
        Pretty print representation of a multicomplex number.

        :param self: A multicomplex number to be converted into a Pretty print
            prepresentation
        :type self: mcomplex

        .. code-block:: Python

            >>> print(mcomplex([1,2,3,4]))
            1 + 2*im(1) + 3*im(2) + 4*im([1, 2])

        ..
        """

        return core.multi_str(self, "im")

    def __mul__(self, other):
        """
        To define how to multiply two multicomplex numbers.
     
        It overloads the multiplication operator "*". It allows
        the multiplication of two multicomplex numbers of different
        orders, no matter how they are sorted, or even the
        multiplication of a multicomplex and a scalar.
     
        :return: mcomplex
        :param self: multicomplex number
        :param other: multicomplex number
        :type self: mcomplex
        :type other: mcomplex

        .. code-block:: Python

           >>> a = mcomplex([1,2,3])
           >>> b = mcomplex([4,5])
           >>> print(a * b)
           -6 + 13*im(1) + 12*im(2) + 15*im([1, 2])

        ..
        """

        coeffs1 = self.coeffs                   # Get attributes from self.
        order1 = self.order
        n1 = self.numcoeffs

        if type(other) in core.scalartypes:     # Case 1. Mult. by real num.

            new = self.__class__(0)             # Create a new mcomplex number.

            new.coeffs = coeffs1 * other        # Overwrite attributes.
            new.order = order1
            new.numcoeffs = n1

        elif isinstance(other, type(self)):

            # Case 2. Mult. by mcomplex.

            new = self.__class__(0)             # Create a new mcomplex number.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            if (n1 >= n2):

                order = order1                  # Set attributes for result.
                n = n1
                p = np.zeros(n, np.float64)     # Allocate space

                core.mcomplex_mul(coeffs1, n1, coeffs2, n2, p)

            else:

                order = order2                  # Set attributes for result.
                n = n2
                p = np.zeros(n, np.float64)     # Allocate space

                core.mcomplex_mul(coeffs2, n2, coeffs1, n1, p)

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

                new_coeffs = np.zeros(shape2 + (n2,), np.float64)
                order = order2
                n = n2

                for i in range(size2):

                    idx = core.item2index(i, shape2, size2)

                    core.mcomplex_mul(coeffs2[idx], n2,
                                      coeffs1, n1, new_coeffs[idx])

            else:

                new_coeffs = np.zeros(shape2 + (n1,), np.float64)
                order = order1
                n = n1

                for i in range(size2):

                    idx = core.item2index(i, shape2, size2)

                    core.mcomplex_mul(
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

        Provides division of an array of multicomplex values with scalars and
        multicomplex numbers.
    
        :param self: multicomplex number
        :param other: scalar or multicomplex number to be divided with self
        :type self: marray
        :type other: mcomplex, int, float

        .. code-block:: Python

           >>> M = marray([[[1,2],[3,4]],
                           [[5,6],[7,8]]])mak
           >>> print(M / M)
           marray([[[1.,0.],  [1.,0.]],
                   [[1.,0.],  [1.,0.]]])
        """

        new = self.__class__(0)                 # Create a new mcomplex number.

        coeffs1 = self.coeffs                   # Get attributes from self.
        order1 = self.order
        n1 = self.numcoeffs

        if type(other) in core.scalartypes:     # Case 1. Div by real number.

            new.coeffs = coeffs1 / other        # Overwrite attributes.
            new.order = order1
            new.numcoeffs = n1

        elif isinstance(other, type(self)):

            # Case 2. Div by multicomplex.

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
            num = np.zeros(n, np.float64)       # Numerator.
            den = np.zeros(n, np.float64)       # Denominator.
            old = np.zeros(n, np.float64)       # Space for copies.
            cden = np.zeros(n, np.float64)      # Conjugated denominator.

            # Initialize numerator and denominator.
            num[:n1] = coeffs1.copy()
            den[:n2] = coeffs2.copy()

            # Compute coefficients of the numerator.
            core.mcomplex_truediv(num, den, old, cden, n, order2)

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
            num = np.zeros(shape + (n,), np.float64) # Numerator.
            den = np.zeros(n, np.float64)            # Denominator.
            old = np.zeros(n, np.float64)            # Space for copies.
            cden = np.zeros(n, np.float64)           # Conjugated denominator.

            for i in range(size):

                idx = core.item2index(i, shape, size)

                # Initialize numerator and denominator.
                num[idx + (slice(0, n1),)] = coeffs1.copy()
                den[:n2] = coeffs2[idx].copy()

                # Compute coefficients of the numerator.
                core.mcomplex_truediv(num[idx], den, old, cden, n, order)

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
        :return: marray
        
        To provide reverse division for multicomplex arrays.

        It defines how to make a division between a scalar and a
        multicomplex array. This function creates a new array
        whose result is the division of the scalar by each
        element of the original array.

        :param self: Real value
        :param other: multicomplex number
        :type self: int, float
        :type other: mcomplex
    
        .. code-block:: Python

            >>> M = marray([[[1,2],[3,4]],
                            [[5,6],[7,8]]])
            >>> print(2 / M)
            marray([[[ 0.4       ,-0.8       ],
                     [ 0.24      ,-0.32      ]],
                    [[ 0.16393443,-0.19672131],
                     [ 0.12389381,-0.14159292]]])
        ..

        """

        new = self.__class__(0)                 # Create a new mcomplex number.
        coeffs = self.coeffs                    # Create a view to self.coeffs.
        order = self.order                      # Get attributes.
        n = self.numcoeffs

        # Allocate space for coefficients.
        num = np.zeros(n, np.float64)           # Numerator.
        old = np.zeros(n, np.float64)           # Space for copies.
        cden = np.zeros(n, np.float64)          # Conjugated denominator.

        num[0] = other                          # Initialize numerator.

        # Allocate and initialize denominator.
        den = coeffs.copy()

        # Compute coefficients of the numerator.
        core.mcomplex_truediv(num, den, old, cden, n, order)

        new.coeffs = num                        # Overwrite attributes.
        new.order = order
        new.numcoeffs = n

        return new

    def __pow__(self, other):
        """
        :return: marray

        Provides the power function method for multicomplex arrays.
        
        This function makes use of the Taylor series expansion approach, which is
        exact for multicomplex numbers.
        
        :param self: multicomplex array
        :param e: multicomplex array to raise the multicomplex array by
        :type self: marray
        :type e: marray
        
        .. code-block:: Python
        
            >>> u = marray([[  1,  2,  3],
                            [  4,  5,  6],
                            [  7,  8,  9],
                            [ 10, 11, 12],
                            [ 13, 14, 15]])
            >>> print(u**2)
            marray([[ -12.,   4.,   6.,  12.],
                    [ -45.,  40.,  48.,  60.],
                    [ -96., 112., 126., 144.],
                    [-165., 220., 240., 264.],
                    [-252., 364., 390., 420.]])
        ..
        """

        if type(other) in core.scalartypes:

            # Create a view to coeffs of x.
            coeffs = self.coeffs
            order = self.order                  # Get attributes.
            n = self.numcoeffs

            new = self.__class__(0)             # Create a new mcomplex number.

            lower = 1                           # Begin Taylor series with 1.

            # Allocate space for coefficients.
            v = np.zeros(n, np.float64)         # Returning value.
            s = np.zeros(n, np.float64)         # s = (z - x0)
            p = np.zeros(n, np.float64)         # Powers of (z - x0). p = s**k.
            old = np.zeros(n, np.float64)       # Space for copies.

            # Initialize variables and arrays.

            # Extract real part from "self".
            x0 = coeffs[0]

            s = coeffs.copy()
            s[0] = 0.
            p[0] = 1.
            v[0] = x0**other

            core.mcomplex_pow(v, s, p, old, x0, other, n, lower, order)

            new.coeffs = v                      # Overwrite attributes.
            new.order = order
            new.numcoeffs = n

        elif isinstance(other, type(self)):

            new = exp(other * log(self))

        return new

    def __rpow__(self, other):
        """
        :return: mcomplex

        To provide reverse power.

        It defines how to raise a real number to a complex power.

        :param self: multicomplex number
        :param other: real number
        :type self: mcomplex
        :type other: int, float

        .. code-block:: Python

            >>> a = mcomplex([1,2,3,4])
            >>> 2**a
            mcomplex([ 2., 2.77258872, 4.15888308,11.31061361])
        ..
        """

        new = exp(self * np.log(other))

        return new

    def get_cr(self, i, j):
        """
        Get value from the position [i,j] of the CR matrix form.

        This is the Cauchy Riemann binary mapping method for a
        multicomplex number. It can retrieve any position of the CR
        matrix.

        :param self: Cauchy-Riemann matrix
        :param i: First index of position
        :param j: Second index of position
        :type self: array-like
        :type i: int
        :type j: int

        .. code-block:: Python

            >>> t = mcomplex([1,8,2,7,3,6,4,5])
            >>> print(t.get_cr(3,4))
            -5

        ..
        """

        return core.mcomplex_get_cr(self.coeffs, i, j)

    def to_cr(self):
        """
        To convert a multicomplex number into its Cauchy-Riemann matrix form.

        :param self: A multicomplex value. Multicomplex values can
           be represented as a scalar, vector, or matrix.
        :type self: mcomplex
        :rtype: array-like

        .. code-block :: Python

            >>> M = marray([[[1,2],[3,4]],
                            [[5,6],[7,8]]])
            >>> M_full = M.to_cr()
            >>> print(mcr_to_marray(M_full, order=1))
            marray([[[1.,2.],  [3.,4.]],
                    [[5.,6.],  [7.,8.]]])
        ..            
        """

        n = self.numcoeffs                      # Get number of coefficients.
        coeffs = self.coeffs                    # Create a view to coeffs of x.
        array = np.zeros([n, n], np.float64)    # Allocate space for CR matrix.

        core.mcomplex_to_cr(coeffs, n, array)   # Fill CR matrix.

        return array


class marray(core.marray):

    def __getitem__(self, index):
        """
        :return: int

        To get the value of a position in an array of multicomplex numbers.

        :param self: multicomplex number
        :param index: position of the desired value
        :type self: mcomplex
        :type index: int, array-like

        .. code-block:: Python

            >>> M = marray([[[1,2],[3,4]],
                            [[5,6],[7,8]]])
            >>> print(M[1,1])
            7 + 8*im(1)
        
        ..
        """

        # Number of dimensions of the array.
        ndims = len(self.shape)

        # Case 1. Index is integer and array is 1D.
        if (type(index) in core.inttypes) and (ndims == 1):

            value = mcomplex(self.coeffs[index])

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

                    value = mcomplex(self.coeffs[index])

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

        This method provides multiplication of an array of multicomplex values with
        scalars and multicomplex numbers.

        :param self: multicomplex number
        :param other: scalar or multicomplex number to be multiplied with self
        :type self: marray
        :type other: mcomplex, int, float

        .. code-block:: Python

            >>> M = marray([[[1,2],[3,4]],
                            [[5,6],[7,8]]])
            >>> print(M * M)
            marray([[[ -3.,  4.],  [ -7., 24.]],
                    [[-11., 60.],  [-15.,112.]]])
        
        """

        new = self.__class__([[0]])             # New array for result.

        coeffs1 = self.coeffs                   # Get mcomplex attributes from
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

        elif isinstance(other, mcomplex):

            # Case 2. Multiplication by mcomplex.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            if (n1 >= n2):

                new_coeffs = np.zeros(shape1 + (n1,), np.float64)
                order = order1
                n = n1

                for i in range(size1):

                    idx = core.item2index(i, shape1, size1)

                    core.mcomplex_mul(coeffs1[idx], n1,
                                      coeffs2, n2, new_coeffs[idx])

            else:

                new_coeffs = np.zeros(shape1 + (n2,), np.float64)
                order = order2
                n = n2

                for i in range(size1):

                    idx = core.item2index(i, shape1, size1)

                    core.mcomplex_mul(
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

                new_coeffs = np.zeros(shape1 + (n1,), np.float64)
                order = order1
                n = n1

                for i in range(size1):

                    idx = core.item2index(i, shape1, size1)

                    core.mcomplex_mul(coeffs1[idx], n1,
                                      coeffs2[idx], n2, new_coeffs[idx])

            else:

                new_coeffs = np.zeros(shape1 + (n2,), np.float64)
                order = order2
                n = n2

                for i in range(size1):

                    idx = core.item2index(i, shape1, size1)

                    core.mcomplex_mul(coeffs2[idx], n2,
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
        
        Provides division of an array of multicomplex values with scalars and
        multicomplex numbers.
        
        :param self: multicomplex number
        :param other: scalar or multicomplex number to be divided with self
        :type self: marray
        :type other: mcomplex, int, float

        .. code-block:: Python

            >>> M = marray([[[1,2],[3,4]],
                            [[5,6],[7,8]]])
            >>> print(M / M)
            marray([[[1.,0.],  [1.,0.]],
                    [[1.,0.],  [1.,0.]]])

        ..
        """

        new = self.__class__([[0]])             # New array for result.

        coeffs1 = self.coeffs                   # Get mcomplex attributes from
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

        elif isinstance(other, mcomplex):

            # Case 2. Division by mcomplex.

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
            num = np.zeros(shape1 + (n,), np.float64)  # Numerator.
            den = np.zeros(n, np.float64)              # Denominator.
            old = np.zeros(n, np.float64)              # Space for copies.
            cden = np.zeros(n, np.float64)             # Conjugated denominator.

            for i in range(size1):

                idx = core.item2index(i, shape1, size1)

                # Initialize numerator and denominator.
                num[idx + (slice(0, n1),)] = coeffs1[idx].copy()
                den[:n2] = coeffs2.copy()

                # Compute coefficients of the numerator.
                core.mcomplex_truediv(num[idx], den, old, cden, n, order2)

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
            num = np.zeros(shape1 + (n,), np.float64)  # Numerator.
            den = np.zeros(n, np.float64)              # Denominator.
            old = np.zeros(n, np.float64)              # Space for copies.
            cden = np.zeros(n, np.float64)             # Conjugated denominator.

            for i in range(size1):

                idx = core.item2index(i, shape1, size1)

                # Initialize numerator and denominator.
                num[idx + (slice(0, n1),)] = coeffs1[idx].copy()
                den[:n2] = coeffs2[idx].copy()

                # Compute coefficients of the numerator.
                core.mcomplex_truediv(num[idx], den, old, cden, n, order)

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

        To provide reverse division for multicomplex arrays.

        It defines how to make a division between a scalar and a
        multicomplex array. This function creates a new array
        whose result is the division of the scalar by each
        element of the original array.

        :type self: int, float ???
        :type other: marray

        .. code-block:: Python

            >>> M = marray([[[1,2],[3,4]],
                            [[5,6],[7,8]]])
            >>> print(2 / M)
            marray([[[ 0.4       ,-0.8       ],
                      [ 0.24      ,-0.32      ]],
                     [[ 0.16393443,-0.19672131],
                      [ 0.12389381,-0.14159292]]])
 
        ..
        """

        new = self.__class__([[0]])               # New array for result.

        coeffs = self.coeffs                      # Create a view to self.coeffs.
        order = self.order                        # Get attributes.
        n = self.numcoeffs

        shape = self.shape                        # Get self's array attributes.
        size = self.size                          # These won't change.

        # Allocate space for coefficients.
        num = np.zeros(shape + (n,), np.float64)  # Numerator.
        den = np.zeros(n, np.float64)             # Denominator.
        old = np.zeros(n, np.float64)             # Space for copies.
        cden = np.zeros(n, np.float64)            # Conjugated denominator.

        for i in range(size):

            idx = core.item2index(i, shape, size)

            # Initialize numerator and denominator.
            num[idx + (0,)] = other
            den = coeffs[idx].copy()

            # Compute coefficients of the numerator.
            core.mcomplex_truediv(num[idx], den, old, cden, n, order)

        new.coeffs = num                        # Overwrite attributes.
        new.order = order
        new.numcoeffs = n

        return new

    def __pow__(self, e):
        """
        :return: marray

        Provides the power function method for multicomplex arrays.

        This function makes use of the Taylor series expansion approach, which is
        exact for multicomplex numbers.

        :param self: multicomplex array
        :param e: multicomplex array to raise the multicomplex array by
        :type self: marray
        :type e: marray

        .. code-block:: Python

            >>> u = marray([[  1,  2,  3],
                            [  4,  5,  6],
                            [  7,  8,  9],
                            [ 10, 11, 12],
                            [ 13, 14, 15]])
            >>> print(u**2)
            marray([[ -12.,   4.,   6.,  12.],
                    [ -45.,  40.,  48.,  60.],
                    [ -96., 112., 126., 144.],
                    [-165., 220., 240., 264.],
                    [-252., 364., 390., 420.]])
        ..
        """

        new = self.__class__([[0]])             # New array for result.

        coeffs = self.coeffs                    # Get mcomplex attributes from
        order = self.order                      # self.
        n = self.numcoeffs

        shape = self.shape                      # Get self's array attributes.
        size = self.size

        lower = 1                               # Begin Taylor series with 1.

        # Allocate space for coefficients.

        # Coefficients of the new array.
        new_coeffs = np.zeros(shape + (n,), np.float64)

        s = np.empty(n, np.float64)             # s = (z - x0)
        p = np.empty(n, np.float64)             # Powers of (z - x0). p = s**k.
        old = np.empty(n, np.float64)           # Space for copies.

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

            core.mcomplex_pow(v, s, p, old, x0, e, n, lower, order)

        new.coeffs = new_coeffs                 # Overwrite attributes.
        new.order = order
        new.numcoeffs = n
        new.shape = shape
        new.size = size

        return new

    def to_cr(self):
        """
        To convert a multicomplex number into its Cauchy-Riemann matrix form.
        
        :param self: A multicomplex. Multicomplex values can
           be represented as a scalar, vector, or matrix.
        
        :type self: mcomplex
        :rtype: array-like

        .. code-block :: Python

            >>> M = marray([[[1,2],[3,4]],
                            [[5,6],[7,8]]])
            >>> M_full = M.to_cr()
            >>> print(mcr_to_marray(M_full, order=1))
            marray([[[1.,2.],  [3.,4.]],
                    [[5.,6.],  [7.,8.]]])
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

            for k in range(numcoeffs):          # For each mcomplex coefficient
                for p in range(size0):          # For each vector position.

                    new[size0 * k + p] = coeffs[p, k]

        elif (ndims == 2) and (shape[0] == shape[1]):

            new = np.zeros((n, n))              # Allocate space.

            for i in range(numcoeffs):          # For each Cauchy-Riemann row.
                for j in range(numcoeffs):      # For each CR column.
                    for p in range(size0):      # For each matrix row.
                        for q in range(size0):  # For each matrix column.

                            new[size0 * i + p, size0 * j + q] = \
                                core.mcomplex_get_cr(coeffs[p, q], i, j)

        else:

            raise IndexError(f"invalid shape {shape}. " +
                             "Array must be 2D square or 1D.")

        return new


def im(imagDirs, order=0):
    """
    To create a multicomplex number with value 1 at the specified imaginary direction.

    :param imagDirs: The imaginary direction of value 1. This location given can
       be given as a scalar or array, depending on the desired location.
    :param order: Order of the multicomplex number
    :type imagDirs: int or array-like
    :type order: int, optional, default=0

    .. code-block :: Python

        >>> print(im(2))
        mcomplex([0, 0, 1, 0])
        >>> print(im[1,2])
        mcomplex([0, 0, 0, 1])
    """

    new = mcomplex(0)                           # New mcomplex number.

    pos = getPos(imagDirs)                      # Get the coefficient position.

    # If order is not defined, compute it.
    if (order == 0):
        order = (pos).bit_length()              # ( = int(ceil(log2(pos+1))) )

    n = 1 << order                              # Compute num. of coefficients.

    new_coeffs = np.zeros(n, np.float64)        # Allocate space for coeffs.

    new_coeffs[pos] = 1.                        # Assign unary value.

    new.coeffs = new_coeffs                     # Overwrite attributes
    new.order = order
    new.numcoeffs = n

    return new


def mzero(order=0):
    """
    To create a multicomplex number of the given order whose coefficients 
        are all zeros.

    :param order: The order of the multicomplex number
    :type order: int, optional, default=0
    """

    new = mcomplex(0)                           # Create a new mcomplex number.
    n = 1 << order                              # Number of coeffs.
    coeffs = np.zeros(n, np.float64)            # Allocate space.

    new.coeffs = coeffs                         # Overwrite information.
    new.order = order
    new.numcoeffs = n

    return new


def zeros(shape, order=0):
    """
    Create an array filled with null multicomplex numbers of the given 
        order.
 
    :param shape: Shape of the container of zeros
    :param order: Order of the container of zeros
    :type shape: int
    :type order: int, optional, default=0

    .. code-block:: Python

       >>> a = mzero(1)
       >>> b = mzero(2)
       >>> a
       0 + 0*im(1)
       >>> b
       0 + 0*im(1) + 0*im(2) + 0*im([1,2])
    ..

    """

    new = marray([[0]])                         # Create a new mcomplex array.
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

    coeffs = np.zeros(np_shape, np.float64)     # Create coeffs array.

    new.coeffs = coeffs                         # Overwrite information.
    new.order = order
    new.numcoeffs = n
    new.shape = new_shape
    new.size = new_size

    return new


def rarray(coef_list, order=0):
    """
    To create an array of multicomplexs from an array of real
    numbers.
      
    The constructor of the array class is not designed to
    parse arrays of real numbers as multicomplexs. Use this
    function to make the conversion.

    :param coef_list: A list of coefficients to be applied to an array of
       multicomplexs
    :param order: Order of the multicomplex array
    :type coef_list: array-like
    :type order: int, optional, default=0

    .. code-block:: Python

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

    new = marray([[0]])                         # Create a new mcomplex array.
    n = 1 << order                              # Number of coeffs per entry.

    shape = core.getshape(coef_list)            # Get array attributes.
    size = np.prod(shape)

    ndims = len(shape)                          # Get its number of dimensions.

    # Allocate space.
    coeffs = np.zeros(shape + (n,), np.float64)

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
    Provide exponential function for multicomplex numbers.
    
    This function is based on the Taylor series expansion
    approach, which is exact for multicomplex numbers.
    
    :param x: Input of the exponential function
    :type x: mcomplex

    .. code-block :: Python
        >>> x = mcomplex([1,2,3,4])
        >>> print(exp(x))
        6.7957 - 27.1828*im(1) - 13.5914*im(2) + 27.1828*im([1, 2])
    ..
    """

    if type(x) in core.scalartypes:

        new = mcomplex(0)                       # Create a new mcomplex number.

        new.coeffs[0] = np.exp(x)

    elif isinstance(x, mcomplex):

        coeffs = x.coeffs                       # Create a view to coeffs of x.
        order = x.order                         # Get attributes.
        n = x.numcoeffs

        lower = 1                               # Begin Taylor series with 1.

        new = mcomplex(0)                       # Create a new mcomplex number.

        # Allocate space for coefficients.
        v = np.zeros(n, np.float64)             # Returning value.
        s = np.zeros(n, np.float64)             # s = (z - x0)
        p = np.zeros(n, np.float64)             # Powers of (z - x0). p = s**k.
        old = np.zeros(n, np.float64)           # Space for copies.

        # Initialize variables and arrays.

        x0 = coeffs[0]                          # Extract self's real part.

        s = coeffs.copy()
        s[0] = 0.
        p[0] = 1.
        v[0] = 1.

        core.mcomplex_exp(v, s, p, old, x0, n, lower, order)

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

        new = marray([[0]])                     # Create a new mcomplex array.

        # Allocate space for coefficients.

        # Coefficients of the new array.
        new_coeffs = np.zeros(shape + (n,), np.float64)

        s = np.empty(n, np.float64)             # s = (z - x0)
        p = np.empty(n, np.float64)             # Powers of (z - x0). p = s**k.
        old = np.empty(n, np.float64)           # Space for copies.

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

            core.mcomplex_exp(v, s, p, old, x0, n, lower, order)

        new.coeffs = new_coeffs                 # Overwrite attributes.
        new.order = order
        new.numcoeffs = n
        new.shape = shape
        new.size = size

    return new


def log(x):
    """
    Provide logarithm function for multicomplex numbers.
    
    Wrapper for the logarithm of the Cauchy Riemann matrix
    version of a multicomplex number.
    
    :param x: Input of the logarithm function
    :type x:  mcomplex

    .. code-block :: Python 
      
        >>> x = mcomplex([1,2,3,4])
        >>> print(log(x))
        -1.5 + 14*im(1) + 11*im(2) - 2*im([1, 2])
    .. 
    """

    if type(x) in core.scalartypes:

        new = mcomplex(0)                       # Create a new mcomplex number.

        new.coeffs[0] = np.log(x)

    elif isinstance(x, mcomplex):

        coeffs = x.coeffs                       # Create a view to coeffs of x.
        order = x.order                         # Get attributes.
        n = x.numcoeffs

        lower = 1                               # Begin Taylor series with 1.

        new = mcomplex(0)                       # Create a new mcomplex number.

        # Allocate space for coefficients.
        v = np.zeros(n, np.float64)             # Returning value.
        s = np.zeros(n, np.float64)             # s = (z - x0)
        p = np.zeros(n, np.float64)             # Powers of (z - x0). p = s**k.
        old = np.zeros(n, np.float64)           # Space for copies.

        # Initialize variables and arrays.

        x0 = coeffs[0]                          # Extract self's real part.

        s = coeffs.copy()
        s[0] = 0.
        p[0] = 1.
        v[0] = np.log(x0)

        core.mcomplex_log(v, s, p, old, x0, n, lower, order)

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

        new = marray([[0]])                     # Create a new mcomplex array.

        # Allocate space for coefficients.

        # Coefficients of the new array.
        new_coeffs = np.zeros(shape + (n,), np.float64)

        s = np.empty(n, np.float64)             # s = (z - x0)
        p = np.empty(n, np.float64)             # Powers of (z - x0). p = s**k.
        old = np.empty(n, np.float64)           # Space for copies.

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

            core.mcomplex_log(v, s, p, old, x0, n, lower, order)

        new.coeffs = new_coeffs                 # Overwrite attributes.
        new.order = order
        new.numcoeffs = n
        new.shape = shape
        new.size = size

    # End if. Variable type cases.

    return new


def sin(x):
    """
    Provide sine function for multicomplex numbers.
    
    This function is based on the Taylor series expansion
    approach, which is exact for multicomplex numbers.
    
    :param x: Input of the sine function
    :type x:  mcomplex

    .. code-block :: Python

        >>> x = mcomplex([1,2,3,4])
        >>> print(sin(x))
        -0.420735 + 11.1783*im(1) + 8.35267*im(2) - 2.88762*im([1, 2])
    ..
    """

    if type(x) in core.scalartypes:

        new = mcomplex(0)                       # Create a new mcomplex number.

        new.coeffs[0] = np.sin(x)

    elif isinstance(x, mcomplex):

        coeffs = x.coeffs                       # Create a view to coeffs of x.
        order = x.order                         # Get attributes.
        n = x.numcoeffs

        lower = 1                               # Begin Taylor series with 1.

        new = mcomplex(0)                       # Create a new mcomplex number.

        # Allocate space for coefficients.
        v = np.zeros(n, np.float64)             # Returning value.
        s = np.zeros(n, np.float64)             # s = (z - x0)
        p = np.zeros(n, np.float64)             # Powers of (z - x0). p = s**k.
        old = np.zeros(n, np.float64)           # Space for copies.

        # Initialize variables and arrays.

        x0 = coeffs[0]                          # Extract self's real part.

        s = coeffs.copy()
        s[0] = 0.
        p[0] = 1.
        v[0] = np.sin(x0)

        core.mcomplex_sin(v, s, p, old, x0, n, lower, order)

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

        new = marray([[0]])                     # Create a new mcomplex array.

        # Allocate space for coefficients.

        # Coefficients of the new array.
        new_coeffs = np.zeros(shape + (n,), np.float64)

        s = np.empty(n, np.float64)             # s = (z - x0)
        p = np.empty(n, np.float64)             # Powers of (z - x0). p = s**k.
        old = np.empty(n, np.float64)           # Space for copies.

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

            core.mcomplex_sin(v, s, p, old, x0, n, lower, order)

        new.coeffs = new_coeffs                 # Overwrite attributes.
        new.order = order
        new.numcoeffs = n
        new.shape = shape
        new.size = size

    # End if. Variable type cases.

    return new


def cos(x):
    """
    Provide cosine function for multicomplex numbers.
   
    This function is based on the Taylor series expansion
    approach, which is exact for multicomplex numbers.
   
    :param x: Input of the cosine function
    :type x: mcomplex
    
    .. code-block :: Python

        >>> x = mcomplex([1,2,3,4])
        >>> print(cos(x))
        -0.270151 + 4.80069*im(1) + 1.79801*im(2) - 6.6077*im([1, 2])
    ..
    """

    if type(x) in core.scalartypes:

        new = mcomplex(0)                       # Create a new mcomplex number.

        new.coeffs[0] = np.cos(x)

    elif isinstance(x, mcomplex):

        coeffs = x.coeffs                       # Create a view to coeffs of x.
        order = x.order                         # Get attributes.
        n = x.numcoeffs

        lower = 1                               # Begin Taylor series with 1.

        new = mcomplex(0)                       # Create a new mcomplex number.

        # Allocate space for coefficients.
        v = np.zeros(n, np.float64)             # Returning value.
        s = np.zeros(n, np.float64)             # s = (z - x0)
        p = np.zeros(n, np.float64)             # Powers of (z - x0). p = s**k.
        old = np.zeros(n, np.float64)           # Space for copies.

        # Initialize variables and arrays.

        x0 = coeffs[0]                          # Extract self's real part.

        s = coeffs.copy()
        s[0] = 0.
        p[0] = 1.
        v[0] = np.cos(x0)

        core.mcomplex_cos(v, s, p, old, x0, n, lower, order)

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

        new = marray([[0]])                     # Create a new mcomplex array.

        # Allocate space for coefficients.

        # Coefficients of the new array.
        new_coeffs = np.zeros(shape + (n,), np.float64)

        s = np.empty(n, np.float64)             # s = (z - x0)
        p = np.empty(n, np.float64)             # Powers of (z - x0). p = s**k.
        old = np.empty(n, np.float64)           # Space for copies.

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

            core.mcomplex_cos(v, s, p, old, x0, n, lower, order)

        new.coeffs = new_coeffs                 # Overwrite attributes.
        new.order = order
        new.numcoeffs = n
        new.shape = shape
        new.size = size

    return new


def sqrt(x):
    """
    Provide sqrt function for multicomplex numbers and arrays.
    
    This function is based on the Taylor series expansion
    approach, which is exact for multicomplex numbers.
    
    :param x: Input of the square root function
    :type x:  mcomplex

    .. code-block :: Python

        >>> x = mcomplex([1,2,3,4])
        >>> print(sqrt(x))
        0.625 + 4*im(1) + 3.5*im(2) + 0.5*im([1, 2])
    ..
    """

    if type(x) in core.scalartypes:

        new = mcomplex(0)                       # Create a new mcomplex number.

        new.coeffs[0] = np.sqrt(x)

    elif isinstance(x, mcomplex):

        coeffs = x.coeffs                       # Create a view to coeffs of x.
        order = x.order                         # Get attributes.
        n = x.numcoeffs

        e0 = 0.5                                # Exponent for square root.
        lower = 1                               # Begin Taylor series with 1.

        new = mcomplex(0)                       # Create a new mcomplex number.

        # Allocate space for coefficients.
        v = np.zeros(n, np.float64)             # Returning value.
        s = np.zeros(n, np.float64)             # s = (z - x0)
        p = np.zeros(n, np.float64)             # Powers of (z - x0). p = s**k.
        old = np.zeros(n, np.float64)           # Space for copies.

        # Initialize variables and arrays.

        x0 = coeffs[0]                          # Extract self's real part.

        s = coeffs.copy()
        s[0] = 0.
        p[0] = 1.
        v[0] = x0**e0

        core.mcomplex_pow(v, s, p, old, x0, e0, n, lower, order)

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

        new = marray([[0]])                     # Create a new mcomplex array.

        # Allocate space for coefficients.

        # Coefficients of the new array.
        new_coeffs = np.zeros(shape + (n,), np.float64)

        s = np.empty(n, np.float64)             # s = (z - x0)
        p = np.empty(n, np.float64)             # Powers of (z - x0). p = s**k.
        old = np.empty(n, np.float64)           # Space for copies.

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

            core.mcomplex_pow(v, s, p, old, x0, e0, n, lower, order)

        new.coeffs = new_coeffs                 # Overwrite attributes.
        new.order = order
        new.numcoeffs = n
        new.shape = shape
        new.size = size

    return new


def dot(array1, array2):
    """
    Dot product for n-dimensional arrays of multicomplex numbers.
 
    :param array1: n-dimensional array of multicomplex numbers
    :param array2: n-dimensional array of multicomplex numbers
    :type array1:  mcomplex
    :type array2:  mcomplex

    .. code-block :: Python

        >>> u = marray([[1,2],
                        [3,4]])
        >>> v = marray([[5,6],
                        [7,8]])
        >>> print(dot(u,v))
        -18 + 68*im(1)
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

        new = mcomplex(0)                       # Create a new mcomplex number.

        if (nc1 >= nc2):                        # Case 1.1.

            order = order1                      # Set attributes for result.
            nc = nc1

            temp = np.zeros((n, nc), np.float64)

            core.mcxarray_mul(coeffs1, nc1, coeffs2, nc2, temp)

            new_coeffs = sum(temp)

        else:                                   # Case 1.2.

            order = order2                      # Set attributes for result.
            nc = nc2

            temp = np.zeros((n, nc), np.float64)

            core.mcxarray_mul(coeffs2, nc2, coeffs1, nc1, temp)

            new_coeffs = sum(temp)

        new.coeffs = new_coeffs                 # Overwrite information.
        new.order = order
        new.numcoeffs = nc

    else:

        # Case 2. Mat-vec, vec-mat, mat-mat.

        new = marray([[0]])                     # Create a new mcomplex array.

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
            new_coeffs = np.zeros(new_shape + (nc,), np.float64)
            temp = np.zeros(nc, np.float64)

            for i in range(psize1):
                for j in range(psize2):

                    idx3 = core.item2index(i * p + j, new_shape, new_size)

                    for k in range(n):

                        idx1 = core.item2index(i * n + k, shape1, size1)
                        idx2 = core.item2index(j + k * p, shape2, size2)

                        core.mcomplex_mul_rs(
                            coeffs1[idx1], nc1, coeffs2[idx2], nc2, temp)

                        new_coeffs[idx3] += temp

        else:                                   # Case 2.2.

            order = order2                      # Set attributes for result.
            nc = nc2

            # Allocate space.
            new_coeffs = np.zeros(new_shape + (nc,), np.float64)
            temp = np.zeros(nc, np.float64)

            for i in range(psize1):
                for j in range(psize2):

                    idx3 = core.item2index(i * p + j, new_shape, new_size)

                    for k in range(n):

                        idx1 = core.item2index(i * n + k, shape1, size1)
                        idx2 = core.item2index(j + k * p, shape2, size2)

                        core.mcomplex_mul_rs(
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
    Dot product for n-dimensional arrays of multicomplex numbers.

    :param array1: n-dimensional array of multicomplex numbers
    :param array2: n-dimensional array of multicomplex numbers
    :type array1:  mcomplex
    :type array2:  mcomplex
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

        new = mcomplex(0)                       # Create a new mcomplex number.

        if (nc1 >= nc2):                        # Case 1.1.

            order = order1                      # Set attributes for result.
            nc = nc1
            new_coeffs = np.zeros(nc, np.float64)
            temp = np.zeros(nc, np.float64)

            for i in range(n):
                core.mcomplex_mul_rs(coeffs1[i], nc1, coeffs2[i], nc2, temp)
                new_coeffs += temp

        else:                                   # Case 1.2.

            order = order2                      # Set attributes for result.
            nc = nc2
            new_coeffs = np.zeros(nc, np.float64)
            temp = np.zeros(nc, np.float64)

            for i in range(n):
                core.mcomplex_mul_rs(coeffs2[i], nc2, coeffs1[i], nc1, temp)
                new_coeffs += temp

        new.coeffs = new_coeffs                 # Overwrite information.
        new.order = order
        new.numcoeffs = nc

    else:

        # Case 2. Mat-vec, vec-mat, mat-mat.

        new = marray([[0]])                     # Create a new mcomplex array.

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
            new_coeffs = np.zeros(new_shape + (nc,), np.float64)
            temp = np.zeros(nc, np.float64)

            for i in range(psize1):
                for j in range(psize2):

                    idx3 = core.item2index(i * p + j, new_shape, new_size)

                    for k in range(n):

                        idx1 = core.item2index(i * n + k, shape1, size1)
                        idx2 = core.item2index(j + k * p, shape2, size2)

                        core.mcomplex_mul_rs(
                            coeffs1[idx1], nc1, coeffs2[idx2], nc2, temp)

                        new_coeffs[idx3] += temp

        else:                                   # Case 2.2.

            order = order2                      # Set attributes for result.
            nc = nc2

            # Allocate space.
            new_coeffs = np.zeros(new_shape + (nc,), np.float64)
            temp = np.zeros(nc, np.float64)

            for i in range(psize1):
                for j in range(psize2):

                    idx3 = core.item2index(i * p + j, new_shape, new_size)

                    for k in range(n):

                        idx1 = core.item2index(i * n + k, shape1, size1)
                        idx2 = core.item2index(j + k * p, shape2, size2)

                        core.mcomplex_mul_rs(
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
    To convert a Cauchy-Riemann matrix to multicomplex number.

    :param x: Cauchy-Riemann matrix
    :type x: array-like
    
    .. code-block:: Python
    
       >>> t = mcomplex([1,8,2,7,3,6,4,5])
       >>> T = t.to_cr()
       >>> print(mcr_to_mnumber(T))
       1 + 8*im(1) + 2*im(2) + 7*im([1, 2]) + 3*im(3) + 6*im([1, 3]) + 4*im([2, 3]) 
          + 5*im([1, 2, 3])
    
    ..
    """

    new = mcomplex(x[:, 0])

    return new


def mcr_to_marray(x, order):
    """
    To convert a Cauchy-Riemann expanded vector or square
    matrix to array of mcomplex.

    :param x: Cauchy-Riemann expanded vector, or square matrix
    :param order: order of square matrix
    :type x: array-like
    :type order: int
    :rtype:  mcomplex array

    .. code-block:: Python
    
       >>> t = mcomplex([1,8,2,7,3,6,4,5])
       >>> T = t.to_cr()
       >>> print(mcr_to_marray(T,1))
       marray([[[ 1., 3.],  [-8.,-6.],  [-2.,-4.],  [ 7., 5.]],
               [[ 8., 6.],  [ 1., 3.],  [-7.,-5.],  [-2.,-4.]],
               [[ 2., 4.],  [-7.,-5.],  [ 1., 3.],  [-8.,-6.]],
               [[ 7., 5.],  [ 2., 4.],  [ 8., 6.],  [ 1., 3.]]])
    
    ..
    """

    new = marray([[0]])

    numcoeffs = 1 << order                      # Number of mcomplex coeffs.

    np_shape = x.shape                          # Shape of the input array.
    ndims = len(np_shape)                       # Num. dimensions of the array.

    # Size (mcomplex) of the 1st dimension.
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

        for k in range(numcoeffs):              # For each mcomplex coefficient
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
