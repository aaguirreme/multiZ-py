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
Module name:  core.

Description:  This module defines the base classes and common code for
              multicomplex and multidual algebras.

Date:         May 5, 2016
Updated:      March 16, 2021
'''

import numpy as np

inttypes = (int, np.int32, np.int64)
floattypes = (float, np.float64)
scalartypes = inttypes + floattypes


class mnumber:

    def __init__(self, coef_list):
        """
        PURPOSE:      Constructor of the superclass number.

        DESCRIPTION:  Creates a new multicomplex or multidual number given its
                      coefficients.

        PARAMETERS
                      coef_list: A list of real coefficients.

        RESULT:       The next attributes are created:
                      obj.coeffs:    a numpy 1D array that contains the
                                    coefficients.
                      obj.order:    the order of the number.
                      obj.numcoeffs: the number of coefficients. Depends on the
                                    order.
        """

        if type(coef_list) in scalartypes:

            # Store attributes.
            self.order = 0
            self.numcoeffs = 1

            # Store the coefficient.
            self.coeffs = np.array([coef_list], np.float64)

        else:

            # Get the original number of coefficients
            n = len(coef_list)

            # Compute attributes.
            order = (n - 1).bit_length()        # ( = int(ceil(log2(n))) )
            numcoeffs = 1 << order              # ( = 2**order )

            # Allocate space for coefficients.
            coeffs = np.zeros(numcoeffs, np.float64)

            # Store coefficients.
            for i in range(n):
                coeffs[i] = coef_list[i]
            # End for.

            # Store attributes
            self.coeffs = coeffs
            self.order = order
            self.numcoeffs = numcoeffs

        # End if.

    def __repr__(self):
        """
        PURPOSE:  To print a representation of the multicomplex or multidual
                  object that could be used to create new objects.
        """

        head = self.__class__.__name__ + "("
        body = np.array2string(self.coeffs, separator=',')
        tail = ')'

        return (head + body + tail)

    def __getitem__(self, index):
        """
        PURPOSE:  To extract a coefficient from the object.
        """

        return self.coeffs[index]

    def __setitem__(self, index, value):
        """
        PURPOSE:  To set the value of a coefficient of the object.
        """

        self.coeffs[index] = value

    def __neg__(self):
        """
        PURPOSE:      To define how to turn an object into its opposite
                      (negative)

        DESCRIPTION:  It overloads the opperator "-".
        """

        new = self.__class__(0)                 # New object for result.

        new.coeffs = -self.coeffs               # Overwrite attributes.
        new.order = self.order
        new.numcoeffs = self.numcoeffs

        return new

    def __add__(self, other):
        """
        PURPOSE:      To define how to sum two objects.

        DESCRIPTION:  It overloads the sum operator "+". It allows the addition
                      of two multicomplex or two multidual numbers of different
                      orders, no matter how they are sorted, or even the
                      addition of an object and a scalar.
        """

        coeffs1 = self.coeffs                   # Get attributes from self.
        order1 = self.order
        n1 = self.numcoeffs

        if type(other) in scalartypes:          # Case 1. Sum to real number.

            new = self.__class__(0)             # New object for result.

            s = coeffs1.copy()
            s[0] += other

            new.coeffs = s                      # Overwrite attributes
            new.order = order1
            new.numcoeffs = n1

        elif isinstance(other, type(self)):

            # Case 2. Sum to object of the same type.

            new = self.__class__(0)             # New object for result.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            if (n1 >= n2):

                s = coeffs1.copy()
                order = order1
                n = n1

                for i in range(n2):
                    s[i] += coeffs2[i]
                # End for.

            else:

                s = coeffs2.copy()
                order = order2
                n = n2

                for i in range(n1):
                    s[i] += coeffs1[i]
                # End for.

            # End if. Order comparison.

            new.coeffs = s                      # Overwrite attributes.
            new.order = order
            new.numcoeffs = n

        elif isinstance(other, marray):         # Case 3. Addition with array.

            new = other.__class__([[0]])        # New array for result.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            # Get array attributes from other.
            shape2 = other.shape
            size2 = other.size

            if (n2 >= n1):

                new_coeffs = coeffs2.copy()
                order = order2
                n = n2

                for i in range(size2):

                    idx = item2index(i, shape2, size2)

                    for j in range(n1):
                        new_coeffs[idx + (j,)] += coeffs1[j]
                    # End for.

                # End for.

            else:

                new_coeffs = np.zeros(shape2 + (n1,), np.float64)
                order = order1
                n = n1

                for i in range(size2):

                    idx = item2index(i, shape2, size2)

                    for j in range(n2):

                        idx_j = idx + (j,)

                        new_coeffs[idx_j] = coeffs1[j] + coeffs2[idx_j]

                    # End for

                    for j in range(n2, n1):
                        new_coeffs[idx + (j,)] = coeffs1[j]
                    # End for.

                # End for.

            # End if. Order comparison.

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order
            new.numcoeffs = n
            new.shape = shape2
            new.size = size2

        else:                                   # Case 4. Incompatible types.

            raise TypeError("unsupported operand type(s) for +: '%s' and '%s'"
                            % (strtype(self), strtype(other)))

        # End if. Type cases.

        return new

    def __radd__(self, other):
        """
        PURPOSE:      To provide reverse addition.

        DESCRIPTION:  It defines how to add an object with a scalar (integer or
                      float) when the scalar value is placed before the
                      addition operator "+".
        """

        new = self.__class__(0)                 # New object for result.

        s = self.coeffs.copy()
        s[0] += other

        new.coeffs = s                          # Overwrite attributes
        new.order = self.order
        new.numcoeffs = self.numcoeffs

        return new

    def __sub__(self, other):
        """
        PURPOSE:      To define how to subtract two objects.

        DESCRIPTION:  It overloads the subtraction operator "-". It allows the
                      subtraction of two multicomplex or two multidual numbers
                      of different orders, no matter how they are sorted, or
                      even the subtraction of an object and a scalar.
        """

        coeffs1 = self.coeffs                   # Get attributes from self.
        order1 = self.order
        n1 = self.numcoeffs

        # Case 1. Subtraction with real num.
        if type(other) in scalartypes:

            new = self.__class__(0)             # New object for result.

            s = coeffs1.copy()
            s[0] -= other

            new.coeffs = s                      # Overwrite attributes
            new.order = order1
            new.numcoeffs = n1

        elif isinstance(other, type(self)):

            # # Case 2. Subtraction with object of the same type.

            new = self.__class__(0)             # New object for result.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            if (n1 >= n2):

                s = coeffs1.copy()
                order = order1
                n = n1

                for i in range(n2):
                    s[i] -= coeffs2[i]
                # End for i.

            else:

                s = -coeffs2
                order = order2
                n = n2

                for i in range(n1):
                    s[i] += coeffs1[i]
                # End for i.

            # End if. Order comparison.

            new.coeffs = s                      # Overwrite attributes.
            new.order = order
            new.numcoeffs = n

        elif isinstance(other, marray):         # Case 3. Sub. with array.

            new = other.__class__([[0]])        # New array for result.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            # Get array attributes from other.
            shape2 = other.shape
            size2 = other.size

            if (n2 >= n1):

                new_coeffs = -coeffs2
                order = order2
                n = n2

                for i in range(size2):

                    idx = item2index(i, shape2, size2)

                    for j in range(n1):
                        new_coeffs[idx + (j,)] += coeffs1[j]
                    # End for.

                # End for.

            else:

                new_coeffs = np.zeros(shape2 + (n1,), np.float64)
                order = order1
                n = n1

                for i in range(size2):

                    idx = item2index(i, shape2, size2)

                    for j in range(n2):

                        idx_j = idx + (j,)

                        new_coeffs[idx_j] = coeffs1[j] - coeffs2[idx_j]

                    # End for

                    for j in range(n2, n1):
                        new_coeffs[idx + (j,)] = coeffs1[j]
                    # End for.

                # End for.

            # End if. Order comparison.

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order
            new.numcoeffs = n
            new.shape = shape2
            new.size = size2

        else:                                   # Case 4. Incompatible types.

            raise TypeError("unsupported operand type(s) for -: '%s' and '%s'"
                            % (strtype(self), strtype(other)))

        # End if. Type cases.

        return new

    def __rsub__(self, other):
        """
        PURPOSE:      To provide reverse subtraction.

        DESCRIPTION:  It defines how to add the negative of an object and a
                      scalar (integer or float) when the scalar value is placed
                      before the subtraction operator "-".
        """

        new = self.__class__(0)                 # New object for result.

        s = -self.coeffs
        s[0] += other

        new.coeffs = s                          # Overwrite attributes
        new.order = self.order
        new.numcoeffs = self.numcoeffs

        return new

    def __rmul__(self, other):
        """
        PURPOSE:      To provide reverse multiplication.

        DESCRIPTION:  It defines how to multiply an object with a scalar
                      (integer or float) when the scalar value is placed before
                      the multiplication operator "*".
        """

        new = self.__class__(0)                 # New object for result.

        new.coeffs = self.coeffs * other          # Overwrite attributes
        new.order = self.order
        new.numcoeffs = self.numcoeffs

        return new

    def real(self):
        """
        PURPOSE:  Extract real part of a number.
        """

        return self.coeffs[0]

    def imag(self, imagDir):
        """
        PURPOSE:  Extract imaginary part of the specified imaginary direction.
        """

        # Get the position from the multiindex.
        pos = getPos(imagDir)

        return self.coeffs[pos]

    def setIm(self, imagDir, value):
        """
        PURPOSE:  Set imaginary part of the specified imaginary direction.
        """

        # Get the position from the multiindex.
        pos = getPos(imagDir)

        self.coeffs[pos] = value

    def copy(self):
        """
        PURPOSE:      To create a copy of the object, totally independent of
                      the original.

        DESCRIPTION:  The equality operator "=" is intended to create an alias
                      or a view of a container. You should use copy() when you
                      need a copy istead of an alias.
        """

        # New object of the current class.
        new = self.__class__(0)

        new.coeffs = self.coeffs.copy()         # Copy all attributes.
        new.order = self.order
        new.numcoeffs = self.numcoeffs

        return new

    def change_order(self, new_order):
        """
        PURPOSE:      To change the order of the object.

        DESCRIPTION:  If the defined order is greater than the current, new
                      positions are filled with zeros. If it is lower, the
                      number of positions will be truncated to match the
                      defined order.
        """

        old_order = self.order                  # Get info from self.
        old_n = self.numcoeffs
        old_coeffs = self.coeffs

        new_n = 1 << new_order                  # Allocate space for
        new_coeffs = np.zeros(new_n, np.float64)  # temp. storage.

        if (new_order >= old_order):
            n = old_n
        else:
            n = new_n
        # End if.

        new_coeffs[:n] = old_coeffs[:n].copy()  # Copy coefficients.

        self.coeffs = new_coeffs                # Overwrite information.
        self.order = new_order
        self.numcoeffs = new_n

    def conjugate(self, imagDir):
        """
        PURPOSE:  To conjugate all the components that have the specified
                  imaginary direction.

        EXAMPLE:  >>> a = mcomplex([1,2,3,4])
                  >>> a.conjugate(1)
                  mcomplex([ 1.,-2., 3.,-4.])
                  >>> a.conjugate(2)
                  mcomplex([ 1., 2.,-3.,-4.])
        """

        new = self.__class__(0)                 # New object for result.

        coeffs = self.coeffs                    # Create a view of coeffs.
        order = self.order                      # Get attributes.
        n = self.numcoeffs
        new_coeffs = np.zeros(n, np.float64)      # Allocate space.

        multi_conjugate(coeffs, n, imagDir, new_coeffs)

        new.coeffs = new_coeffs                 # Overwrite attributes
        new.order = order
        new.numcoeffs = n

        return new

    def tovector(self):
        """
        PURPOSE:  To convert a object to numpy's vector.

        EXAMPLE:  >>> a = mcomplex([1,2,3,4])
                  >>> a.tovector()
                  array([ 1.,  2.,  3.,  4.])
        """

        return self.coeffs.copy()

    def tolist(self):
        """
        PURPOSE:  To extract the coefficients of an object into a list.

        EXAMPLE:  >>> a = mcomplex([1,2,3,4])
                  >>> a.tolist()
                  [ 1, 2, 3, 4])
        """

        return self.coeffs.tolist()


class marray:

    def __init__(self, coef_list):
        """
        PURPOSE:      Constructor of the superclass array.

        DESCRIPTION:  Creates a new N-dimensional array to store multicomplex
                      or multidual numbers.

        PARAMETERS
                      coef_list:  A numpy array or nested list containing the
                                  values of the coefficients.

        RESULT:       The next attributes are created:
                      obj.coeffs:    a numpy array that contains the
                                    coefficients.
                      obj.order:    the order of any number stored. Integer
                      obj.numcoeffs: the number of coefficients of any number
                                    stored. Integer.
                      obj.shape:    Shape of the object array.
                      obj.size:     Total number of stored numbers.
        """

        # Get dimensions of the original array.
        np_shape = getshape(coef_list)
        np_size = np.prod(np_shape)

        ndims = len(np_shape)                   # Get its number of dimensions.

        # Split shape of full array.
        n = np_shape[ndims - 1]                 # Orig. number of coefficients.
        shape = np_shape[:ndims - 1]            # Shape of the object array.
        size = np.prod(shape)                   # Size of the object array.

        # Compute attributes of stored object (mcomplex or mdual).
        order = (n - 1).bit_length()            # ( = int(ceil(log2(n))) )
        numcoeffs = 1 << order                  # ( = 2**order )

        # Allocate space for coefficients.
        coeffs = np.zeros(shape + (numcoeffs,), np.float64)

        # Store coefficients.
        for i in range(np_size):

            # Get index.
            idx = item2index(i, np_shape, np_size)

            # Initialize value. View to coef_list.
            value = coef_list[idx[0]]

            for j in range(1, ndims):
                value = value[idx[j]]           # Reduce window to coef_list.
            # End for.

            coeffs[idx] = value

        # End for.

        # Store attributes
        self.coeffs = coeffs
        self.order = order
        self.numcoeffs = numcoeffs
        self.shape = shape
        self.size = size

    def __repr__(self):
        """
        PURPOSE:      To print a representation of the array that could be used
                      to create new arrays.

        DESCRIPTION:  If the matrix has more than 100 positions in any of its
                      axis, the output will be reduced.
        """

        head = self.__class__.__name__ + '('
        body = np.array2string(self.coeffs, separator=',', prefix=head)

        if (len(self.shape) == 2):              # Apply for 2D arrays only.
            body = body.replace(',\n', ',')
            body = body.replace('        ', ' ')
        # End if.

        tail = ')'

        return (head + body + tail)

    def __setitem__(self, index, value):
        """
        PURPOSE:  To set the value of a position in an array.
        """

        # Number of dimensions of the array.
        ndims = len(self.shape)

        # Case 1. Index is integer and array is 1D.
        if (type(index) in inttypes) and (ndims == 1):

            self.__setpos(index, value)

        # Case 2. Index is a tuple.
        elif isinstance(index, tuple):

            nidx = len(index)                   # Number of positions in index.

            if any(i is Ellipsis for i in index):

                raise IndexError("indices not valid for array")

            elif (nidx <= ndims + 1):

                all_int = all(type(i) in inttypes for i in index)

                if (nidx == ndims + 1) and all_int:

                    self.coeffs[index] = value

                elif (nidx == ndims) and all_int:

                    self.__setpos(index, value)

                elif (nidx <= ndims):

                    self.__setslice(index, value)

                else:

                    raise IndexError("indices not valid for array")

                # End if.

            else:

                raise IndexError("too many indices for array")

            # End if. Tuple cases.

        # Case 3. Other cases.
        else:

            self.__setslice(index, value)

        # End if.

    def __getslice(self, index):
        """
        PURPOSE:  To get a slice of an array made of mcomplex or mdual numbers.
        """

        value = self.__class__([[0]])           # Create a wrapper.
        coeffs = self.coeffs[index]             # Create a view to the slice.
        order = self.order                      # Get element attributes.
        numcoeffs = self.numcoeffs

        ndims = len(coeffs.shape)               # Num. of dimensions of coeffs.
        shape = coeffs.shape[:ndims - 1]        # Compute array attributes.
        size = coeffs.size // numcoeffs

        value.coeffs = coeffs                   # Overwrite attributes.
        value.order = order
        value.numcoeffs = numcoeffs
        value.shape = shape
        value.size = size

        return value

    def __setpos(self, index, value):
        """
        PURPOSE:  To set value to a single position of an array.
        """

        n1 = self.numcoeffs                     # Get numcoeffs of self.

        shape = self.shape                      # Get array attributes.
        size = self.size

        try:                                    # Convert index to tuple.
            idx = tuple(index)
        except TypeError:
            idx = tuple([index])
        # End if.

        # Case 1. Assign a real value to position.
        if type(value) in scalartypes:

            self.coeffs[idx] = np.zeros(n1)     # Reset values of the position.

            # Assign value to the real part.
            self.coeffs[idx + (0,)] = value

        # Case 2. Assign mcomplex/mdual to position.
        else:

            order2 = value.order                # Get attributes of value.
            n2 = value.numcoeffs

            if (n1 > n2):                       # If value has lower order
                # reset values of the position.
                self.coeffs[idx] = np.zeros(n1)

            elif (n1 < n2):                     # If value has a higher order
                # change order of self.

                coeffs = self.coeffs            # View of coeffs.

                temp = np.zeros(shape + (n2,), np.float64)

                # Copy coeffs to temp.
                array_copy(coeffs, size, shape, n1, temp)

                self.coeffs = temp              # Overwrite information.
                self.order = order2
                self.numcoeffs = n2

            # End if.

            coeffs1 = self.coeffs[idx]          # Create views of the position
            coeffs2 = value.coeffs              # and the value.
            coeffs1[:n2] = coeffs2.copy()       # Copy coefficients.

        # End if.

    def __setslice(self, index, value):
        """
        PURPOSE:  To set value to a slice of an array.
        """

        coeffs1 = self.coeffs[index]            # Create a view to the slice.
        coeffs2 = value.coeffs                  # and to value.

        np_shape1 = coeffs1.shape               # Get the shape of the slice.
        ndims = len(np_shape1)
        shape1 = np_shape1[:ndims - 1]

        shape2 = value.shape                    # Get array attributes of
        size2 = value.size                      # Value.

        if (shape1 != shape2):                  # Verify if shapes are equal.
            raise ValueError(
                'could not broadcast input array from shape %s into shape %s'
                % (shape2, shape1))
        # End if.

        coeffs1 = np.array([0])                 # Deactivate view.

        n1 = self.numcoeffs                     # Get number of coefficients
        n2 = value.numcoeffs                    # of self and value.

        order2 = value.order                    # Get order of value.

        if (n1 > n2):                           # If value has lower order
            # reset values in slice.
            self.coeffs[index] = np.zeros(np_shape1)

        elif (n1 < n2):                         # If value has a higher order
            # change order of self.

            # View of coeffs of the whole array.
            coeffs = self.coeffs
            shape = self.shape                  # Get array attributes.
            size = self.size

            temp = np.zeros(shape + (n2,), np.float64)

            # Copy coeffs to temp.
            array_copy(coeffs, size, shape, n1, temp)

            self.coeffs = temp                  # Overwrite information.
            self.order = order2
            self.numcoeffs = n2

        # End if.

        coeffs1 = self.coeffs[index]            # Reactivate view.

        # Copy coefficients of the slice.
        array_copy(coeffs2, size2, shape2, n2, coeffs1)

    def __neg__(self):
        """
        PURPOSE:      To define how to turn an array into its opposite
                      (negative)

        DESCRIPTION:  It overloads the opperator "-".
        """

        new = self.__class__([[0]])             # New array for result.

        new.coeffs = -self.coeffs               # Overwrite attributes.
        new.order = self.order
        new.numcoeffs = self.numcoeffs
        new.shape = self.shape
        new.size = self.size

        return new

    def __add__(self, other):
        """
        PURPOSE:      to provide addition between an array and other types.

        DESCRIPTION:  this method provides addition of an array of mcomplex or
                      mdual with:
                      - Scalars.
                      - Multicomplex numbers.
                      - Other arrays.
        """

        new = self.__class__([[0]])             # New array for result.

        # Get mcomplex attributes from self.
        coeffs1 = self.coeffs
        order1 = self.order
        n1 = self.numcoeffs

        shape1 = self.shape                     # Array attributes of self.
        size1 = self.size                       # These won't change.

        if type(other) in scalartypes:

            # Case 1. Addition with scalars.

            new_coeffs = coeffs1.copy()

            for i in range(size1):

                idx = item2index(i, shape1, size1)

                new_coeffs[idx + (0,)] += other

            # End for.

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order1
            new.numcoeffs = n1
            new.shape = shape1
            new.size = size1

        elif isinstance(other, mnumber):

            # Case 2. Addition with mcomplex number or mdual number.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            if (n1 >= n2):

                new_coeffs = coeffs1.copy()
                order = order1
                n = n1

                for i in range(size1):

                    idx = item2index(i, shape1, size1)

                    for j in range(n2):
                        new_coeffs[idx + (j,)] += coeffs2[j]
                    # End for.

                # End for.

            else:

                new_coeffs = np.zeros(shape1 + (n2,), np.float64)
                order = order2
                n = n2

                for i in range(size1):

                    idx = item2index(i, shape1, size1)

                    for j in range(n1):

                        idx_j = idx + (j,)

                        new_coeffs[idx_j] = coeffs1[idx_j] + coeffs2[j]

                    # End for

                    for j in range(n1, n2):
                        new_coeffs[idx + (j,)] = coeffs2[j]
                    # End for.

                # End for.

            # End if. Order comparison.

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order
            new.numcoeffs = n
            new.shape = shape1
            new.size = size1

        elif isinstance(other, type(self)):

            # Case 3. Addition with an array.

            # Get array attributes from other.
            shape2 = other.shape
            size2 = other.size

            if (shape1 != shape2):
                raise ValueError("operands could not be broadcast together " +
                                 "with shapes %s %s" % (shape1, shape2))
            # End if.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            if (n1 == n2):

                order = order1
                n = n1

                new_coeffs = coeffs1 + coeffs2

            elif (n1 > n2):

                order = order1
                n = n1

                new_coeffs = coeffs1.copy()
                new_coeffs[..., :n2] += coeffs2

            else:

                order = order2
                n = n2

                new_coeffs = coeffs2.copy()
                new_coeffs[..., :n1] += coeffs1

            # End if. Order comparison.

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order
            new.numcoeffs = n
            new.shape = shape1
            new.size = size1

        else:

            # Case 4. Incompatible types.

            raise TypeError("unsupported operand type(s) for +: '%s' and '%s'"
                            % (strtype(self), strtype(other)))

        # End if. Type cases.

        return new

    def old_add(self, other):
        """
        PURPOSE:      to provide addition between an array and other types.

        DESCRIPTION:  this method provides addition of an array of mcomplex or
                      mdual with:
                      - Scalars.
                      - Multicomplex numbers.
                      - Other arrays.
        """

        new = self.__class__([[0]])             # New array for result.

        # Get mcomplex attributes from self.
        coeffs1 = self.coeffs
        order1 = self.order
        n1 = self.numcoeffs

        shape1 = self.shape                     # Array attributes from self.
        size1 = self.size                       # These won't change.

        if type(other) in scalartypes:

            # Case 1. Addition with scalars.

            new_coeffs = coeffs1.copy()

            for i in range(size1):

                idx = item2index(i, shape1, size1)

                new_coeffs[idx + (0,)] += other

            # End for.

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order1
            new.numcoeffs = n1
            new.shape = shape1
            new.size = size1

        elif isinstance(other, mnumber):

            # Case 2. Addition with mcomplex number or mdual number.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            if (n1 >= n2):

                new_coeffs = coeffs1.copy()
                order = order1
                n = n1

                for i in range(size1):

                    idx = item2index(i, shape1, size1)

                    for j in range(n2):
                        new_coeffs[idx + (j,)] += coeffs2[j]
                    # End for.

                # End for.

            else:

                new_coeffs = np.zeros(shape1 + (n2,), np.float64)
                order = order2
                n = n2

                for i in range(size1):

                    idx = item2index(i, shape1, size1)

                    for j in range(n1):

                        idx_j = idx + (j,)

                        new_coeffs[idx_j] = coeffs1[idx_j] + coeffs2[j]

                    # End for

                    for j in range(n1, n2):
                        new_coeffs[idx + (j,)] = coeffs2[j]
                    # End for.

                # End for.

            # End if. Order comparison.

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order
            new.numcoeffs = n
            new.shape = shape1
            new.size = size1

        elif isinstance(other, type(self)):

            # Case 3. Addition with array.

            # Get array attributes from other.
            shape2 = other.shape
            size2 = other.size

            if (shape1 != shape2):
                raise ValueError("operands could not be broadcast together " +
                                 "with shapes %s %s" % (shape1, shape2))
            # End if.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            if (n1 >= n2):

                new_coeffs = coeffs1.copy()
                order = order1
                n = n1

                for i in range(size1):

                    idx = item2index(i, shape1, size1)

                    for j in range(n2):

                        idx_j = idx + (j,)

                        new_coeffs[idx_j] += coeffs2[idx_j]

                    # End for.

                # End for.

            else:

                new_coeffs = np.zeros(shape1 + (n2,), np.float64)
                order = order2
                n = n2

                for i in range(size1):

                    idx = item2index(i, shape1, size1)

                    for j in range(n1):

                        idx_j = idx + (j,)

                        new_coeffs[idx_j] = coeffs1[idx_j] + coeffs2[idx_j]

                    # End for

                    for j in range(n1, n2):

                        idx_j = idx + (j,)

                        new_coeffs[idx_j] = coeffs2[idx_j]

                    # End for.

                # End for.

            # End if. Order comparison.

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order
            new.numcoeffs = n
            new.shape = shape1
            new.size = size1

        else:

            # Case 4. Incompatible types.

            raise TypeError("unsupported operand type(s) for +: '%s' and '%s'"
                            % (strtype(self), strtype(other)))

        # End if. Type cases.

        return new

    def __radd__(self, other):
        """
        PURPOSE:  To provide reverse addition to array.
        """

        new = self.__class__([[0]])             # New array for result.

        # Get mcomplex attributes from self.
        coeffs1 = self.coeffs
        order1 = self.order
        n1 = self.numcoeffs

        shape1 = self.shape                     # Get self's array attributes.
        size1 = self.size                       # These won't change.

        new_coeffs = coeffs1.copy()

        for i in range(size1):

            idx = item2index(i, shape1, size1)

            new_coeffs[idx + (0,)] += other

        # End for.

        new.coeffs = new_coeffs                 # Overwrite attributes.
        new.order = order1
        new.numcoeffs = n1
        new.shape = shape1
        new.size = size1

        return new

    def __sub__(self, other):
        """
        PURPOSE:      to provide subtraction between an array and other types.

        DESCRIPTION:  this method provides subtraction of an array of mcomplex
                      or mdual with:
                      - Scalars.
                      - Multicomplex numbers.
                      - Other arrays.
        """

        new = self.__class__([[0]])             # New array for result.

        # Get mcomplex attributes from self.
        coeffs1 = self.coeffs
        order1 = self.order
        n1 = self.numcoeffs

        shape1 = self.shape                     # Get self's array attributes.
        size1 = self.size                       # These won't change.

        if type(other) in scalartypes:

            # Case 1. Subtraction with scalars.

            new_coeffs = coeffs1.copy()

            for i in range(size1):

                idx = item2index(i, shape1, size1)

                new_coeffs[idx + (0,)] -= other

            # End for.

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order1
            new.numcoeffs = n1
            new.shape = shape1
            new.size = size1

        elif isinstance(other, mnumber):

            # Case 2. Subtraction with mcomplex or mdual.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            if (n1 >= n2):

                new_coeffs = coeffs1.copy()
                order = order1
                n = n1

                for i in range(size1):

                    idx = item2index(i, shape1, size1)

                    for j in range(n2):
                        new_coeffs[idx + (j,)] -= coeffs2[j]
                    # End for.

                # End for.

            else:

                new_coeffs = np.zeros(shape1 + (n2,), np.float64)
                order = order2
                n = n2

                for i in range(size1):

                    idx = item2index(i, shape1, size1)

                    for j in range(n1):

                        idx_j = idx + (j,)

                        new_coeffs[idx_j] = coeffs1[idx_j] - coeffs2[j]

                    # End for

                    for j in range(n1, n2):
                        new_coeffs[idx + (j,)] = -coeffs2[j]
                    # End for.

                # End for.

            # End if. Order comparison.

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order
            new.numcoeffs = n
            new.shape = shape1
            new.size = size1

        elif isinstance(other, type(self)):

            # Case 3. Subtraction with array.

            # Get array attributes from other.
            shape2 = other.shape
            size2 = other.size

            if (shape1 != shape2):
                raise ValueError("operands could not be broadcast together " +
                                 "with shapes %s %s" % (shape1, shape2))
            # End if.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            if (n1 == n2):

                order = order1
                n = n1

                new_coeffs = coeffs1 - coeffs2

            elif (n1 > n2):

                order = order1
                n = n1

                new_coeffs = coeffs1.copy()
                new_coeffs[..., :n2] -= coeffs2

            else:

                order = order2
                n = n2

                new_coeffs = -coeffs2
                new_coeffs[..., :n1] += coeffs1

            # End if. Order comparison.

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order
            new.numcoeffs = n
            new.shape = shape1
            new.size = size1

        else:                                 # Case 4. Incompatible types.

            raise TypeError("unsupported operand type(s) for +: '%s' and '%s'"
                            % (strtype(self), strtype(other)))

        # End if. Type cases.

        return new

    def old_sub(self, other):
        """
        PURPOSE:      to provide subtraction between an array and other types.

        DESCRIPTION:  this method provides subtraction of an array of mcomplex
                      or mdual with:
                      - Scalars.
                      - Multicomplex numbers.
                      - Other arrays.
        """

        new = self.__class__([[0]])             # New array for result.

        # Get mcomplex attributes from self.
        coeffs1 = self.coeffs
        order1 = self.order
        n1 = self.numcoeffs

        shape1 = self.shape                     # Array attributes from self.
        size1 = self.size                       # These won't change.

        if type(other) in scalartypes:

            # Case 1. Subtraction with scalars.

            new_coeffs = coeffs1.copy()

            for i in range(size1):

                idx = item2index(i, shape1, size1)

                new_coeffs[idx + (0,)] -= other

            # End for.

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order1
            new.numcoeffs = n1
            new.shape = shape1
            new.size = size1

        elif isinstance(other, mnumber):

            # Case 2. Subtraction with mcomplex or mdual.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            if (n1 >= n2):

                new_coeffs = coeffs1.copy()
                order = order1
                n = n1

                for i in range(size1):

                    idx = item2index(i, shape1, size1)

                    for j in range(n2):
                        new_coeffs[idx + (j,)] -= coeffs2[j]
                    # End for.

                # End for.

            else:

                new_coeffs = np.zeros(shape1 + (n2,), np.float64)
                order = order2
                n = n2

                for i in range(size1):

                    idx = item2index(i, shape1, size1)

                    for j in range(n1):

                        idx_j = idx + (j,)

                        new_coeffs[idx_j] = coeffs1[idx_j] - coeffs2[j]

                    # End for

                    for j in range(n1, n2):
                        new_coeffs[idx + (j,)] = -coeffs2[j]
                    # End for.

                # End for.

            # End if. Order comparison.

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order
            new.numcoeffs = n
            new.shape = shape1
            new.size = size1

        elif isinstance(other, type(self)):

            # Case 3. Subtraction with array.

            # Get array attributes from other.
            shape2 = other.shape
            size2 = other.size

            if (shape1 != shape2):
                raise ValueError("operands could not be broadcast together " +
                                 "with shapes %s %s" % (shape1, shape2))
            # End if.

            coeffs2 = other.coeffs              # Get attributes from other.
            order2 = other.order
            n2 = other.numcoeffs

            if (n1 >= n2):

                new_coeffs = coeffs1.copy()
                order = order1
                n = n1

                for i in range(size1):

                    idx = item2index(i, shape1, size1)

                    for j in range(n2):

                        idx_j = idx + (j,)

                        new_coeffs[idx_j] -= coeffs2[idx_j]

                    # End for.

                # End for.

            else:

                new_coeffs = np.zeros(shape1 + (n2,), np.float64)
                order = order2
                n = n2

                for i in range(size1):

                    idx = item2index(i, shape1, size1)

                    for j in range(n1):

                        idx_j = idx + (j,)

                        new_coeffs[idx_j] = coeffs1[idx_j] - coeffs2[idx_j]

                    # End for

                    for j in range(n1, n2):

                        idx_j = idx + (j,)

                        new_coeffs[idx_j] = -coeffs2[idx_j]

                    # End for.

                # End for.

            # End if. Order comparison.

            new.coeffs = new_coeffs             # Overwrite attributes.
            new.order = order
            new.numcoeffs = n
            new.shape = shape1
            new.size = size1

        else:

            # Case 4. Incompatible types.

            raise TypeError("unsupported operand type(s) for +: '%s' and '%s'"
                            % (strtype(self), strtype(other)))

        # End if. Type cases.

        return new

    def __rsub__(self, other):
        """
        PURPOSE:  To provide reverse subtraction to array.
        """

        new = self.__class__([[0]])             # New array for result.

        # Get mcomplex attributes from self.
        coeffs1 = self.coeffs
        order1 = self.order
        n1 = self.numcoeffs

        shape1 = self.shape                     # Array attributes from self.
        size1 = self.size                       # These won't change.

        new_coeffs = -coeffs1

        for i in range(size1):

            idx = item2index(i, shape1, size1)

            new_coeffs[idx + (0,)] += other

        # End for.

        new.coeffs = new_coeffs                 # Overwrite attributes.
        new.order = order1
        new.numcoeffs = n1
        new.shape = shape1
        new.size = size1

        return new

    def __rmul__(self, other):
        """
        PURPOSE:  To provide reverse multiplication to array.
        """

        new = self.__class__([[0]])             # New array for result.

        # Get mcomplex attributes from self.
        coeffs1 = self.coeffs
        order1 = self.order
        n1 = self.numcoeffs

        shape1 = self.shape                     # Get self's array attributes.
        size1 = self.size                       # These won't change.

        new_coeffs = coeffs1.copy()

        new_coeffs *= other

        new.coeffs = new_coeffs                 # Overwrite attributes.
        new.order = order1
        new.numcoeffs = n1
        new.shape = shape1
        new.size = size1

        return new

    def getCoef(self, pos):
        """
        PURPOSE:  Create a view to the values of the given coefficient only.
        """

        if (type(pos) in inttypes):

            view = self.coeffs[..., pos]

        else:

            raise TypeError('unrecognized input for position')

        # End if.

        return view

    def setCoef(self, pos, value):
        """
        PURPOSE:  To set the values of the given coefficient only.
        """

        if (type(pos) in inttypes):

            self.coeffs[..., pos] = value

        else:

            raise TypeError('unrecognized input for position')

        # End if.

    def real(self):
        """
        PURPOSE:  Create a copy of the values of the real part of the array.
        """

        return self.coeffs[..., 0].copy()

    def imag(self, imagDir):
        """
        PURPOSE:  Create a copy of the values of the given imaginary directions
                  only.
        """

        # Get the position from the multiindex.
        pos = getPos(imagDir)

        return self.coeffs[..., pos].copy()

    def setIm(self, imagDir, value):
        """
        PURPOSE:  Create a view to the values of the given imaginary directions
                  only.
        """

        # Get the position from the multiindex.
        pos = getPos(imagDir)

        self.coeffs[..., pos] = value

    def copy(self):
        """
        PURPOSE:      To create a copy of an array, totally independent of the
                      original.

        DESCRIPTION:  The equality operator "=" is intended to create aliases
                      of arrays. You should use copy() when you need a copy
                      istead of an alias.
        """

        new = self.__class__([[0]])

        new.coeffs = self.coeffs.copy()         # Copy all attributes.
        new.order = self.order
        new.numcoeffs = self.numcoeffs
        new.size = self.size
        new.shape = self.shape

        return new

    def change_order(self, new_order):
        """
        PURPOSE:      To change the order of the array.

        DESCRIPTION:  If the defined order is greater than the current, new
                      positions are filled with zeros. If it is lower, the
                      number of positions will be truncated to match the
                      defined order.
        """

        old_coeffs = self.coeffs                # Get mcomplex or mdual
        old_order = self.order                  # old attributes.
        old_n = self.numcoeffs

        size = self.size                        # Get array attributes.
        shape = self.shape                      # These won't be modified.

        # Allocate space for temp. storage.
        new_n = 1 << new_order
        new_coeffs = np.zeros(shape + (new_n,), np.float64)

        if (new_order >= old_order):            # Minimum number of
            n = old_n                           # coefficients to copy.
        else:
            n = new_n
        # End if.

        array_copy(old_coeffs, size, shape, n, new_coeffs)

        self.coeffs = new_coeffs                # Overwrite information.
        self.order = new_order
        self.numcoeffs = new_n

    def transpose(self, axes=None):
        """
        PURPOSE:      Returns a view of the array with axes transposed.

        DESCRIPTION:  By default, reverse the dimensions, otherwise permute the
                      axes according to the values given in axes.

        PARAMETERS:
                      axes: list of ints, optional.
        """

        return transpose(self, axes)

    def T(self, axes=None):
        """
        PURPOSE:      Returns a view of the array with axes transposed.

        DESCRIPTION:  By default, reverse the dimensions, otherwise permute the
                      axes according to the values given in axes.

        PARAMETERS:
                      axes: list of ints, optional.
        """

        return transpose(self, axes)


def getImagDirs(pos):
    """
    PURPOSE:      To convert the position of a multicomplex or multidual
                  coefficient to a list of its imaginary directions.

    DESCRIPTION:  The sorting order of the coefficients of a mcomplex or
                  mdual number and its imaginary directions are related
                  through binary numeral system.

                  The conversion of the position of a coefficient to binary
                  number gives a string that indicates the presence or
                  absence of an imaginary direction. This string must be read
                  backwards to determine the imaginary direction number, i.e,
                  whether it is i[1], i[2], etc.

                  Taking into account that the real part of a mcomplex or
                  mdual number is the zeroth coefficient, look for example a
                  bicomplex number a0 + a1*i[1] + a_2*i[2] + a_3*i[1]*i[2].

                  bin(0) --> 00 --> Real part.
                  bin(1) --> 01 --> Direction i[1]
                  bin(2) --> 10 --> Direction i[2]
                  bin(3) --> 11 --> Directions i[1] and i[2]

    PARAMETERS
                  pos: an integer. The position of a coefficient in the
                       number.

    RESULT:       A list of indices of imaginary direction is returned.

    EXAMPLE:      For a tricomplex number or above order mcx number, get
                  the imaginary directions of the 5th coefficient.

                  T = a_0 + a_1*i[1] + a_2*i[2] + a_3*i[1]*i[2] + a_4*i[3] +
                      a_5*i[1]*i[3] + a_6*i[2]*i[3] + a_7*i[1]*i[2]*i[3]

                  >>> getImagDirs(5)
                  [1, 3]

    NOTE:         In Python3, this implementation is 7 times faster than
                  decomposing the input into powers of 2 through np.log2, and
                  is also 16% more efficient than version 2, which uses
                  bitwise operations.
    """

    # Convert position to binary number, and trim the "0b" prefix.
    string = bin(pos)[2:]

    n = len(string)                             # Get the size of the string

    dirs = []   # Initialize list of imaginary directions.

    for i in range(1, n + 1):                   # Fill the list with
        if (string[n - i] == '1'):              # the imaginary
            dirs.append(i)                      # directions.
        # End if.
    # End for.

    return dirs


def getImagDirsV2(pos):
    """
    PURPOSE:      To convert the position of a mcx coefficient to a list
                  of its imaginary directions.

    DESCRIPTION:  see getImagDirs.

    PARAMETERS
                  pos: an integer. The position of a coefficient in the
                       number.

    RESULT:       A list of indices of imaginary direction is returned.

    EXAMPLE:      For a tricomplex number or above order mcx number, get
                  the imaginary directions of the 5th coefficient.

                  T = a_0 + a_1*i[1] + a_2*i[2] + a_3*i[1]*i[2] + a_4*i[3] +
                      a_5*i[1]*i[3] + a_6*i[2]*i[3] + a_7*i[1]*i[2]*i[3]

                  >>> getImagDirs(5)
                  [1, 3]
    """

    # Initialize list of imaginary directions.
    dirs = []

    # Position "pos" is compared each cycle with a test number that has 1 in
    # one of its digits, e.g, 0b0100.

    # test: comparison number.
    # d:    digit counter for pure imaginary directions.
    test = 1
    d = 1

    while (pos >= test):

        # Stop when the test number is greater than pos.

        if (pos & test):                        # & is bitwise AND.
            dirs.append(d)                      # If comparison is not zero,
        # End if.                               # add direction to list.

        d += 1

        test = (1 << d - 1)                     # << is left shift operator.

    # End while

    return dirs


def getPos(imagDirs):
    """
    PURPOSE:      To convert a list of imaginary directions to coefficient
                  position of a multicomplex or multidual number.

    DESCRIPTION:  The sorting order of the coefficients of a multicomplex (or
                  multidual) and its imaginary directions are related through
                  binary numeral system.

                  Imaginary directions can be interpreted as the digits of a
                  binary number having value "1". The position of the
                  mcx coefficient that involves those directions can be
                  obtained summing powers of 2 of those directions.

    PARAMETERS
                  imagDirs: The list of imaginary directions. List of ints.

    RESULT:       An integer. The position of the coefficient that involves
                  the given directions.

    EXAMPLE:      For a tricomplex or above order mcx number, get the
                  position of the coefficient that involves imaginary
                  directions 1 and 3.

                  T = a_0 + a_1*i[1] + a_2*i[2] + a_3*i[1]*i[2] + a_4*i[3] +
                      a_5*i[1]*i[3] + a_6*i[2]*i[3] + a_7*i[1]*i[2]*i[3]

                  >>> getPos([1,3])
                  5
    """

    pos = 0

    try:

        ndirs = len(imagDirs)

    except TypeError:

        if (imagDirs > 0):
            pos = 1 << (imagDirs - 1)
        # End if.

    else:

        # Note: 1 << (k-1) is the bitwise equiv. of 2**k.

        for k in imagDirs:
            if (k > 0):
                pos += 1 << (k - 1)
            # End if.
        # End for.

    # End try.

    return pos


def transpose(x, axes=None):
    """
    PURPOSE:      Permute dimensions of the array.

    DESCRIPTION:  By default, reverse the dimensions, otherwise permute the
                  axes according to the values given in axes.

                  A transposed view of x is returned whenever is possible.

    PARAMETERS
                  x:    array of mcomplex numbers.
                  axes: list of ints, optional.
    """

    new = x.__class__([[0]])                    # Create same class array.

    old_coeffs = x.coeffs                       # Get old attributes.
    old_shape = x.shape

    order = x.order                             # Get other attributes.
    n = x.numcoeffs
    size = x.size

    ndims = len(old_shape)                      # Get number of dimensions.

    if (axes is None):

        # Reverse all dimensions.
        axes = np.arange(ndims - 1, -1, -1)

    elif (len(axes) != ndims):

        raise ValueError("axes don't match array")

    # End if.

    # Get axes of the internal numpy array. Last dimension is reserved for
    # mcomplex coefficients. It must not be swaped.

    np_axes = list(axes) + [ndims]

    new_coeffs = np.transpose(old_coeffs, np_axes)

    new_shape = new_coeffs.shape[:ndims]        # Get new shape.

    new.coeffs = new_coeffs                     # Overwrite information.
    new.order = order
    new.numcoeffs = n
    new.shape = new_shape
    new.size = size

    return new


def multi_str(obj, imstr):
    """
    PURPOSE:      Pretty print for multicomplex or multidual.

    DESCRIPTION:  Function internally used by the __str__ method of the
                  multicomplex and multidual classes.

    PARAMETERS
                  obj:    A multicomplex or multidual object.
                  imstr:  String to identify imaginary directions. It can be
                          "im" for multicomplex and "eps" for multidual
                          numbers.
    """

    coeffs = obj.coeffs                         # Create a view of coeffs.
    n = obj.numcoeffs                           # Get number of coefficients.

    # Initialize string using the real part.
    string = "%g" % (coeffs[0])

    for i in range(1, n):

        # Get the sign of the coefficient.
        s = str("%+g" % coeffs[i])[0]

        imagDirs = getImagDirs(i)               # Get its imaginary directions.

        if (len(imagDirs) == 1):
            dirs = imagDirs[0]
        else:
            dirs = imagDirs
        # End if.

        string += " %s %g*%s(%s)" % (s, abs(coeffs[i]), imstr, dirs)

    return string


def array_copy(old, size, shape, n, new):
    """
    PURPOSE:      To create a copy of an array, totally independent of the
                  original.

    DESCRIPTION:  This function is internally used by the array classes of
                  the mcomplex and mdual modules.

    PARAMETERS
                  old:    Array of coefficients of the original array.
                          np.array.
                  size:   Size attribute of the original array. Integer.
                  shape:  Shape attribute of the original array. Tuple.
                  n:      Number of mcomplex or mdual coefficients of the
                          original array.  integer
                  new:    Array of coefficients of the copy. np.array.

    RESULT:       The values in "old" are copied to "new".
    """

    # Compute numpy attributes.
    np_size = size * n
    np_shape = shape + (n,)

    for k in range(np_size):

        idx = item2index(k, np_shape, np_size)

        new[idx] = old[idx]

    # End for.


def multi_conjugate(orig, n, imagDir, copy):
    """
    PURPOSE:      To conjugate all the components that have the specified
                  imaginary direction.

    DESCRIPTION:  This function is internally used by the classes mcomplex
                  and mdual.
    """

    step = 1 << imagDir  # Step size.
    half = step // 2       # Half step size.

    for i in range(0, n, step):

        for j in range(i, i + half):
            copy[j] = orig[j]
        # End for j

        for j in range(i + half, i + step):
            copy[j] = -orig[j]
        # End for j

    # End for i


def mcomplex_get_cr(coeffs, i, j):
    """
    PURPOSE:      get value from the position [i,j] of the CR matrix form of
                  a multicomplex number.

    DESCRIPTION:  This function is internally used by the mcomplex class.

                  This is the Cauchy Riemann binary mapping method for a
                  multicomplex number. It can retrieve any position of the CR
                  matrix.

    PARAMETERS
                  coeffs:  Array of coefficients of a mcx number. 1D np.array.
                  i:      Row of the Cauchy Riemann matrix. Integer.
                  j:      Column of the Cauchy Riemann matrix.

    RESULT:       The requested position of the Cauchy Riemann matrix is
                  returned by value.
    """

    # k: position of the multicomplex coefficients array to read.

    k = (i ^ j)                                 # ^ is bitwise XOR.

    # Count matching imaginary directions.
    c = bin(j & k).count("1")                   # & is bitwise AND.

    if (c & 1):                                 # If c is an odd number
        value = -coeffs[k]                      # change coeffs sign.
    else:
        value = coeffs[k]
    # End if.

    return value


def mcxarray_get_cr(coeffs, i, j):
    """
    PURPOSE:      get values from positions [i,j] of the CR matrix form of a
                  multicomplex array.

    DESCRIPTION:  This function is internally used by the mcomplex class.

                  This is the Cauchy Riemann binary mapping method for a
                  multicomplex array. It can retrieve any position of the CR
                  matrix.

    PARAMETERS
                  coeffs: Array of coefficients of a mcx array. n-dim np.array.
                  i:      Row of the Cauchy Riemann matrix. Integer.
                  j:      Column of the Cauchy Riemann matrix.

    RESULT:       The requested position of the Cauchy Riemann matrix is
                  returned by value.
    """

    # k: position of the multicomplex coefficients array to read.

    k = (i ^ j)                                 # ^ is bitwise XOR.

    # Count matching imaginary directions.
    c = bin(j & k).count("1")                   # & is bitwise AND.

    if (c & 1):                                 # If c is an odd number
        value = -coeffs[..., k]                 # change coeffs sign.
    else:
        value = coeffs[..., k]
    # End if.

    return value


def mdual_get_cr(coeffs, i, j):
    """
    PURPOSE:      get value from the position [i,j] of the CR matrix form of
                  a multidual number.

    DESCRIPTION:  This function is internally used by the mdual class.

                  This is the Cauchy Riemann binary mapping method for a
                  multidual number. It can retrieve any position of the CR
                  matrix.

    PARAMETERS
                  coeffs:  Array of coefficients of a hdual number. 1D
                          np.array.
                  i:      Row of the Cauchy Riemann matrix. Integer.
                  j:      Column of the Cauchy Riemann matrix.

    RESULT:       The requested position of the Cauchy Riemann matrix is
                  returned by value.
    """

    # k: position of the multidual coefficients array to read.

    k = (i ^ j)                                 # ^ is bitwise XOR.

    # Count matching imaginary directions.
    c = bin(j & k).count("1")                   # & is bitwise AND.

    if (c == 0):
        value = coeffs[k]
    else:
        value = 0.
    # End if.

    return value


def mduarray_get_cr(coeffs, i, j):
    """
    PURPOSE:      get value from the position [i,j] of the CR matrix form of
                  a multidual array.

    DESCRIPTION:  This function is internally used by the mdual class.

                  This is the Cauchy Riemann binary mapping method for a
                  multidual array. It can retrieve any position of the CR
                  matrix.

    PARAMETERS
                  coeffs: Array of coefficients of a mdual array. n-dim
                          np.array.
                  i:      Row of the Cauchy Riemann matrix. Integer.
                  j:      Column of the Cauchy Riemann matrix.

    RESULT:       The requested position of the Cauchy Riemann matrix is
                  returned by value.
    """

    # k: position of the multidual coefficients array to read.

    k = (i ^ j)                                 # ^ is bitwise XOR.

    # Count matching imaginary directions.
    c = bin(j & k).count("1")                   # & is bitwise AND.

    if (c == 0):
        value = coeffs[..., k]
    else:
        value = np.zeros_like(coeffs[..., k])
    # End if.

    return value


def mcomplex_mul(a, n_a, b, n_b, p):
    """
    PURPOSE:      To define how to multiply two multicomplex numbers.

    DESCRIPTION:  This function is internally used by the mcomplex class.

    PARAMETERS
                  a:    Array of coefficients of the mcx number "a". 1D
                        np.array.
                  n_a:  Number of coefficients of the mcx number "a".
                  b:    Array of coefficients of the mcx number "b". 1D
                        np.array.
                  n_b:  Number of coefficients of the mcx number "b".
                  p:    Array of coefficients to store the product,
                        previously initialized with zero values. 1D np.array.

    RESULT:       The product is stored in p.
    """

    for i in range(n_a):
        for j in range(n_b):
            p[i] += mcomplex_get_cr(a, i, j) * b[j]
        # End for j.
    # End for i.


def mcomplex_mul_rs(a, n_a, b, n_b, p):
    """
    PURPOSE:      To define how to multiply two multicomplex numbers.

    DESCRIPTION:  This function is internally used by the mcomplex class. The
                  additional "rs" stands for "reset", cause this function
                  assigns a zero to each coefficient before the
                  multiplication.

    PARAMETERS
                  a:    Array of coefficients of the mcx number "a". 1D
                        np.array.
                  n_a:  Number of coefficients of the mcx number "a".
                  b:    Array of coefficients of the mcx number "b". 1D
                        np.array.
                  n_b:  Number of coefficients of the mcx number "b".
                  p:    Array of coefficients to store the product, not
                        initialized previously. Their values are reseted
                        internally. 1D np.array.

    RESULT:       The product is stored in p.
    """

    p[:n_a] = 0.

    for i in range(n_a):
        for j in range(n_b):
            p[i] += mcomplex_get_cr(a, i, j) * b[j]
        # End for j.
    # End for i.


def mcxarray_mul(a, n_a, b, n_b, p):
    """
    PURPOSE:      To define how to multiply two multicomplex arrays
                  element-wise.

    DESCRIPTION:  This function is internally used by the mcomplex class.

    PARAMETERS
                  a:    Array of coefficients of the multicomplex array "a".
                        n-dim np.array.
                  n_a:  Number of coefficients of the multicomplex array "a".
                  b:    Array of coefficients of the multicomplex array "b".
                        n-dim np.array.
                  n_b:  Number of coefficients of the multicomplex array "b".
                  p:    Array of coefficients to store the product,
                        previously initialized with zero values. n-dim
                        np.array.

    RESULT:       The product is stored in p.
    """

    for i in range(n_a):
        for j in range(n_b):
            p[..., i] += mcxarray_get_cr(a, i, j) * b[..., j]
        # End for j.
    # End for i.


def mdual_mul(a, n_a, b, n_b, p):
    """
    PURPOSE:      To define how to multiply two multidual numbers.

    DESCRIPTION:  This function is internally used by the mdual class.

    PARAMETERS
                  a:    Array of coefficients of the multidual number "a".
                        1D np.array.
                  n_a:  Number of coefficients of the multidual number "a".
                  b:    Array of coefficients of the multidual number "b".
                        1D np.array.
                  n_b:  Number of coefficients of the multidual number "b".
                  p:    Array of coefficients to store the product,
                        previously initialized with zero values. 1D np.array.

    RESULT:       The product is stored in p.
    """

    for i in range(n_a):
        for j in range(min(i + 1, n_b)):
            p[i] += mdual_get_cr(a, i, j) * b[j]
        # End for j.
    # End for i.


def mduarray_mul(a, n_a, b, n_b, p):
    """
    PURPOSE:      To define how to multiply two multidual arrays element-wise.

    DESCRIPTION:  This function is internally used by the mdual class.

    PARAMETERS
                  a:    Array of coefficients of the multidual array "a".
                        n-dim np.array.
                  n_a:  Number of coefficients of the multidual array "a".
                  b:    Array of coefficients of the multidual array "b".
                        n-dim np.array.
                  n_b:  Number of coefficients of the multidual array "b".
                  p:    Array of coefficients to store the product,
                        previously initialized with zero values. n-dim
                        np.array.

    RESULT:       The product is stored in p.
    """

    for i in range(n_a):
        for j in range(min(i + 1, n_b)):
            p[..., i] += mduarray_get_cr(a, i, j) * b[..., j]
        # End for j.
    # End for i.


def mdual_mul_rs(a, n_a, b, n_b, p):
    """
    PURPOSE:      To define how to multiply two multidual numbers.

    DESCRIPTION:  This function is internally used by the mdual class. The
                  additional "rs" stands for "reset", cause this function
                  assigns a zero to each coefficient before the
                  multiplication.

    PARAMETERS
                  a:    Array of coefficients of the multidual number "a".
                        1D np.array.
                  n_a:  Number of coefficients of the multidual number "a".
                  b:    Array of coefficients of the multidual number "b".
                        1D np.array.
                  n_b:  Number of coefficients of the multidual number "b".
                  p:    Array of coefficients to store the product, not
                        initialized previously. Their values are reseted
                        internally. 1D np.array.

    RESULT:       The product is stored in p.
    """

    p[:n_a] = 0.

    for i in range(n_a):
        for j in range(min(i + 1, n_b)):
            p[i] += mdual_get_cr(a, i, j) * b[j]
        # End for j.
    # End for i.


def mcomplex_truediv(num, den, old, cden, n1, order2):
    """
    PURPOSE:      To define how to divide two multicomplex numbers.

    DESCRIPTION:  This function is internally used by the mcomplex class.

    PARAMETERS
                  num:    Coefficients of the numerator. 1D np.array.
                  den:    Coefficients of the denominator. 1D np.array.
                  old:    Space for old coefficients. 1D np.array.
                  cden:   Space for coefficients of the conjugated
                          denominator.  1D np.array.
                  n1:     Number of coefficients of the numerator.
                  order2: Multicomplex order of the denominator.

    RESULT:       The result is stored in num.
    """

    for k in range(order2, 0, -1):

        # Compute number of coefficients of the denominator.
        n2 = 1 << k

        # Conjugate denominator wrt imaginary direction k.
        multi_conjugate(den, n2, k, cden)

        old[:n1] = num[:n1].copy()              # Compute new numerator.
        mcomplex_mul_rs(old, n1, cden, n2, num)

        old[:n2] = den[:n2].copy()              # Compute new denominator.
        mcomplex_mul_rs(old, n2, cden, n2, den)

    # End for.

    for i in range(n1):
        num[i] = num[i] / den[0]                # Compute numerator coeffs.
    # End for.


def mdual_truediv(num, den, old, cden, n1, order2):
    """
    PURPOSE:      To define how to divide two multidual numbers.

    DESCRIPTION:  This function is internally used by the mdual class.

    PARAMETERS
                  num:    Coefficients of the numerator. 1D np.array.
                  den:    Coefficients of the denominator. 1D np.array.
                  old:    Space for old coefficients. 1D np.array.
                  cden:   Space for coefficients of the conjugated
                          denominator.  1D np.array.
                  n1:     Number of coefficients of the numerator.
                  order2: Multidual order of the denominator.

    RESULT:       The result is stored in num.
    """

    for k in range(order2, 0, -1):

        # Compute number of coefficients of the denominator.
        n2 = 1 << k

        # Conjugate denominator wrt imaginary direction k.
        multi_conjugate(den, n2, k, cden)

        old[:n1] = num[:n1].copy()
        mdual_mul_rs(old, n1, cden, n2, num)    # Compute new numerator.

        old[:n2] = den[:n2].copy()
        mdual_mul_rs(old, n2, cden, n2, den)    # Compute new denominator.

    # End for.

    for i in range(n1):
        num[i] = num[i] / den[0]                # Compute numerator coeffs.
    # End for.


def mcomplex_exp(v, s, p, old, x0, nc, lower, upper):
    """
    PURPOSE:        Compute the multicomplex exponential function through the
                    Taylor series approach.

    DESCRIPTION:    This function is internally used by the mcomplex class.

    PARAMETERS:
                v       Initial coefficients for the returning value.
                s       Coefficients for (z - x0), where z is the original
                        multicomplex number and x0 its real part.
                p       Initial coefficients for the k-th power of (z - x0)
                old     Allocated space for copies.
                x0      Real evaluation point for the function.
                nc      Number of coefficients of the multicomplex number.
                lower   Lower bound. The first term of the Taylor series that
                        needs to be added to the returning value.
                upper   Upper bound. The last term of the Taylor series that
                        needs to be added to the returning value.
    """

    # k: iteration index for Taylor series.
    # r: Factorial(k).
    # d: k-th erivative of sin(x) at x = x0.

    r = 1

    #          (k)
    #         f   (x0)          k
    # v = v + --------- (z - x0)
    #             k!

    for k in range(lower, upper + 1):

        r = r * k                               # Update factorial value.

        old = p.copy()                          # p = (z - x0)**k
        mcomplex_mul_rs(old, nc, s, nc, p)

        v += p / r

    # End for.

    v *= np.exp(x0)


def mdual_exp(v, s, p, old, x0, nc, lower, upper):
    """
    PURPOSE:        Compute the multidual exponential function through the
                    Taylor series approach.

    DESCRIPTION:    This function is internally used by the mdual class.

    PARAMETERS:
                v       Initial coefficients for the returning value.
                s       Coefficients for (z - x0), where z is the original
                        multicomplex number and x0 its real part.
                p       Initial coefficients for the k-th power of (z - x0)
                old     Allocated space for copies.
                x0      Real evaluation point for the function.
                nc      Number of coefficients of the multicomplex number.
                lower   Lower bound. The first term of the Taylor series that
                        needs to be added to the returning value.
                upper   Upper bound. The last term of the Taylor series that
                        needs to be added to the returning value.
    """

    # k: iteration index for Taylor series.
    # r: Factorial(k).
    # d: k-th erivative of sin(x) at x = x0.

    r = 1

    #          (k)
    #         f   (x0)          k
    # v = v + --------- (z - x0)
    #             k!

    for k in range(lower, upper + 1):

        r = r * k                               # Update factorial value.

        old = p.copy()                          # p = (z - x0)**k
        mdual_mul_rs(old, nc, s, nc, p)

        v += p / r

    # End for.

    v *= np.exp(x0)


def mcomplex_sin(v, s, p, old, x0, nc, lower, upper):
    """
    PURPOSE:        Compute the multicomplex sine through the Taylor series
                    approach.

    DESCRIPTION:    This function is internally used by the mcomplex class.

    PARAMETERS:
                v       Initial coefficients for the returning value.
                s       Coefficients for (z - x0), where z is the original
                        multicomplex number and x0 its real part.
                p       Initial coefficients for the k-th power of (z - x0)
                old     Allocated space for copies.
                x0      Real evaluation point for the function.
                nc      Number of coefficients of the multicomplex number.
                lower   Lower bound. The first term of the Taylor series that
                        needs to be added to the returning value.
                upper   Upper bound. The last term of the Taylor series that
                        needs to be added to the returning value.
    """

    # k: iteration index for Taylor series.
    # r: Factorial(k).
    # d: k-th erivative of sin(x) at x = x0.

    r = 1

    flist = np.array([np.sin(x0), np.cos(x0)])

    #          (k)
    #         f   (x0)          k
    # v = v + --------- (z - x0)
    #             k!

    for k in range(lower, upper + 1):

        r = r * k                               # Update factorial value.

        ds = 1 - (k & 2)                        # Derivative sign.
        d = ds * flist[k & 1]                   # Derivative value.

        old = p.copy()                          # p = (z - x0)**k
        mcomplex_mul_rs(old, nc, s, nc, p)

        v += d * p / r

    # End for.


def mdual_sin(v, s, p, old, x0, nc, lower, upper):
    """
    PURPOSE:        Compute the multidual sine through the Taylor series
                    approach.

    DESCRIPTION:    This function is internally used by the mdual class.

    PARAMETERS:
                v       Initial coefficients for the returning value.
                s       Coefficients for (z - x0), where z is the original
                        multicomplex number and x0 its real part.
                p       Initial coefficients for the k-th power of (z - x0)
                old     Allocated space for copies.
                x0      Real evaluation point for the function.
                nc      Number of coefficients of the multicomplex number.
                lower   Lower bound. The first term of the Taylor series that
                        needs to be added to the returning value.
                upper   Upper bound. The last term of the Taylor series that
                        needs to be added to the returning value.
    """

    # k: iteration index for Taylor series.
    # r: Factorial(k).
    # d: k-th erivative of sin(x) at x = x0.

    r = 1

    flist = np.array([np.sin(x0), np.cos(x0)])

    #          (k)
    #         f   (x0)          k
    # v = v + --------- (z - x0)
    #             k!

    for k in range(lower, upper + 1):

        r = r * k                               # Update factorial value.

        ds = 1 - (k & 2)
        d = ds * flist[k & 1]

        old = p.copy()
        mdual_mul_rs(old, nc, s, nc, p)

        v += d * p / r

    # End for.


def mcomplex_cos(v, s, p, old, x0, nc, lower, upper):
    """
    PURPOSE:      Compute the multicomplex cosine through the Taylor series
                  approach.

    DESCRIPTION:  This function is internally used by the mcomplex class.

    PARAMETERS:
                v       Initial coefficients for the returning value.
                s       Coefficients for (z - x0), where z is the original
                        multicomplex number and x0 its real part.
                p       Initial coefficients for the k-th power of (z - x0)
                old     Allocated space for copies.
                x0      Real evaluation point for the function.
                nc      Number of coefficients of the multicomplex number.
                lower   Lower bound. The first term of the Taylor series that
                        needs to be added to the returning value.
                upper   Upper bound. The last term of the Taylor series that
                        needs to be added to the returning value.
    """

    # k: iteration index for Taylor series.
    # r: Factorial(k).
    # d: k-th erivative of sin(x) at x = x0.

    r = 1

    flist = np.array([np.sin(x0), np.cos(x0)])

    #          (k)
    #         f   (x0)          k
    # v = v + --------- (z - x0)
    #             k!

    for k in range(lower, upper + 1):

        r = r * k                               # Update factorial value.

        ds = 1 - ((k + 1) & 2)                  # Derivative sign.
        d = ds * flist[(k + 1) & 1]             # Derivative value.

        old = p.copy()
        mcomplex_mul_rs(old, nc, s, nc, p)

        v += d * p / r

    # End for.


def mdual_cos(v, s, p, old, x0, nc, lower, upper):
    """
    PURPOSE:      Compute the multidual cosine through the Taylor series
                  approach.

    DESCRIPTION:  This function is internally used by the mdual class.

    PARAMETERS:
                v       Initial coefficients for the returning value.
                s       Coefficients for (z - x0), where z is the original
                        multidual number and x0 its real part.
                p       Initial coefficients for the k-th power of (z - x0)
                old     Allocated space for copies.
                x0      Real evaluation point for the function.
                nc      Number of coefficients of the multidual number.
                lower   Lower bound. The first term of the Taylor series that
                        needs to be added to the returning value.
                upper   Upper bound. The last term of the Taylor series that
                        needs to be added to the returning value.
    """

    # k: iteration index for Taylor series.
    # r: Factorial(k).
    # d: k-th erivative of sin(x) at x = x0.

    r = 1   # k! = 1  when k = 0.

    flist = np.array([np.sin(x0), np.cos(x0)])

    #          (k)
    #         f   (x0)          k
    # v = v + --------- (z - x0)
    #             k!

    for k in range(lower, upper + 1):

        r = r * k                               # Update factorial value.

        ds = 1 - ((k + 1) & 2)                  # Derivative sign.
        d = ds * flist[(k + 1) & 1]             # Derivative value.

        old = p.copy()
        mdual_mul_rs(old, nc, s, nc, p)

        v += d * p / r

    # End for.


def mcomplex_pow(v, s, p, old, x0, e0, nc, lower, upper):
    """
    PURPOSE:        Compute the multicomplex potential function through the
                    Taylor series approach.

    DESCRIPTION:    This function is internally used by the mcomplex class.

    PARAMETERS:
                v       Initial coefficients for the returning value.
                s       Coefficients for (z - x0), where z is the original
                        multicomplex number and x0 its real part.
                p       Initial coefficients for the k-th power of (z - x0)
                old     Allocated space for copies.
                x0      Real evaluation point for the function.
                e0      Initial exponent.
                nc      Number of coefficients of the multicomplex number.
                lower   Lower bound. The first term of the Taylor series that
                        needs to be added to the returning value.
                upper   Upper bound. The last term of the Taylor series that
                        needs to be added to the returning value.
    """

    # k: iteration index for Taylor series.
    # r: Factorial(k).
    # d: k-th erivative of x**e at x = x0.  d = c*x0**e
    # c: Real coefficient of d.
    # e: Real exponent of d.

    # Initialize variables.

    r = 1
    c = 1.  # Initial coefficient of d.
    e = e0  # Initial exponent.

    #          (k)
    #         f   (x0)          k
    # v = v + --------- (z - x0)
    #             k!

    for k in range(lower, upper + 1):

        r = r * k                               # Update factorial value.

        # Compute new derivative.
        c = c * e                               # Coefficient.
        e = e - 1.                              # Exponent.
        d = c * x0**e                           # Derivative.

        old = p.copy()                          # p = (z - x0)**k
        mcomplex_mul_rs(old, nc, s, nc, p)

        v += d * p / r

    # End for.


def mdual_pow(v, s, p, old, x0, e0, nc, lower, upper):
    """
    PURPOSE:      Compute the multidual potential function through the Taylor
                  series approach.

    DESCRIPTION:  This function is internally used by the mdual class.

    PARAMETERS:
                v       Initial coefficients for the returning value.
                s       Coefficients for (z - x0), where z is the original
                        multidual number and x0 its real part.
                p       Initial coefficients for the k-th power of (z - x0)
                old     Allocated space for copies.
                x0      Real evaluation point for the function.
                e0      Initial exponent.
                nc      Number of coefficients of the multidual number.
                lower   Lower bound. The first term of the Taylor series that
                        needs to be added to the returning value.
                upper   Upper bound. The last term of the Taylor series that
                        needs to be added to the returning value.
    """

    # k: iteration index for Taylor series.
    # r: Factorial(k).
    # d: k-th erivative of x**e at x = x0.  d = c*x0**e
    # c: Real coefficient of d.
    # e: Real exponent of d.

    # Initialize variables.

    r = 1
    c = 1.  # Initial coefficient of d.
    e = e0  # Initial exponent.

    #          (k)
    #         f   (x0)          k
    # v = v + --------- (z - x0)
    #             k!

    for k in range(lower, upper + 1):

        r = r * k                               # Update factorial value.

        # Compute new derivative.
        c = c * e                               # Coefficient.
        e = e - 1.                              # Exponent.
        d = c * x0**e                           # Derivative.

        old = p.copy()                          # p = (z - x0)**k
        mdual_mul_rs(old, nc, s, nc, p)

        v += d * p / r

    # End for.


def mcomplex_log(v, s, p, old, x0, nc, lower, upper):
    """
    PURPOSE:      Compute the multicomplex logarithm function through the
                  Taylor series approach.

    DESCRIPTION:  This function is internally used by the mcomplex class.

    PARAMETERS:
                v       Initial coefficients for the returning value.
                s       Coefficients for (z - x0), where z is the original
                        multicomplex number and x0 its real part.
                p       Initial coefficients for the k-th power of (z - x0)
                old     Allocated space for copies.
                x0      Real evaluation point for the function.
                e0      Initial exponent.
                nc      Number of coefficients of the multicomplex number.
                lower   Lower bound. The first term of the Taylor series that
                        needs to be added to the returning value.
                upper   Upper bound. The last term of the Taylor series that
                        needs to be added to the returning value.
    """

    # k: iteration index for Taylor series.
    # d: k-th erivative of log(x) at x = x0.

    #          (k)
    #         f   (x0)          k
    # v = v + --------- (z - x0)
    #             k!

    for k in range(lower, upper + 1):

        # Compute new derivative (factorial already included)

        ds = -((k - 1) & 1) + (k & 1)           # Sign.
        d = ds / (k * x0**k)                    # Derivative.

        old = p.copy()                          # p = (z - x0)**k
        mcomplex_mul_rs(old, nc, s, nc, p)

        v += d * p

    # End for.


def mdual_log(v, s, p, old, x0, nc, lower, upper):
    """
    PURPOSE:      Compute the multidual logarithm function through the Taylor
                  series approach.

    DESCRIPTION:  This function is internally used by the mdual class.

    PARAMETERS:
                v       Initial coefficients for the returning value.
                s       Coefficients for (z - x0), where z is the original
                        multidual number and x0 its real part.
                p       Initial coefficients for the k-th power of (z - x0)
                old     Allocated space for copies.
                x0      Real evaluation point for the function.
                nc      Number of coefficients of the multidual number.
                lower   Lower bound. The first term of the Taylor series that
                        needs to be added to the returning value.
                upper   Upper bound. The last term of the Taylor series that
                        needs to be added to the returning value.
    """

    # k: iteration index for Taylor series.
    # d: k-th erivative of log(x) at x = x0.

    #          (k)
    #         f   (x0)          k
    # v = v + --------- (z - x0)
    #             k!

    for k in range(lower, upper + 1):

        # Compute new derivative (factorial already included)

        ds = -((k - 1) & 1) + (k & 1)           # Sign.
        d = ds / (k * x0**k)                    # Derivative.

        old = p.copy()                          # p = (z - x0)**k
        mdual_mul_rs(old, nc, s, nc, p)

        v += d * p

    # End for.


def mcomplex_to_cr(coeffs, n, crmat):
    """
    PURPOSE:      To convert a multicomplex number into its Cauchy Riemann
                  matrix form.

    DESCRIPTION:  This function is internally used by the mcomplex class.

    PARAMETERS
                  coeffs:  Array of coefficients of a mcx number. 1D np.array.
                  n:      Number of coefficients. Integer.
                  crmat:  Allocated space for Cauchy Riemann matrix. 2D
                          np.array, size n x n.

    RESULT:       The positions of the CR matrix are stored in array.
    """

    for i in range(n):
        for j in range(n):
            crmat[i, j] = mcomplex_get_cr(coeffs, i, j)
        # End for j.
    # End for i.


def mdual_to_cr(coeffs, n, crmat):
    """
    PURPOSE:      To convert a multidual number into its Cauchy Riemann
                  matrix form.

    DESCRIPTION:  This function is internally used by the mdual class.

    PARAMETERS
                  coeffs:  Array of coefficients of a multidual number. 1D
                          np.array.
                  n:      Number of coefficients. Integer.
                  crmat:  Allocated space for Cauchy Riemann matrix. 2D
                          np.array, size n x n.

    RESULT:       The positions of the CR matrix are stored in array.
    """

    # Note: the CR upper triangular of a mdual is zero.

    for i in range(n):
        for j in range(i + 1):
            crmat[i, j] = mdual_get_cr(coeffs, i, j)
        # End for j.
    # End for i.


def getshape(array_like):
    """
    PURPOSE:      Get the shape of a nested list or array.
    """

    x = array_like                              # Create a view of array like

    shape = []                                  # Init. shape as empty list.

    while True:

        # For each dimension in coeffs (quantity unknown), try to get the
        # length of each dimension.

        try:

            # Add length to the list. When done, it raises an error.
            shape += [len(x)]

            # If no error, make a smaller view of array_like.
            x = x[0]

        except TypeError:

            break

        # End try block

    # End while.

    return tuple(shape)


def item2index(item, shape, size):
    """
    PURPOSE:      Map the item identificator of an N-dimensional array with
                  its index (coordinates).

    DESCRIPTION:  This function helps to navigate an N-dimensional numpy
                  array, where making nested "for" loops is not always a
                  feasible option.

    PARAMETERS
                  item:   Identificator of an array element. It can be any
                          integer number from zero to size-1.
                  shape:  Shape of the array. Tuple containing the size of
                          each dimension.
                  size:   Size of the array. Total number of stored elements.

    RESULT:       index is returned by value. A tuple containing the
                  coordinates of the element.
    """

    ndims = len(shape)                          # Number of array dimensions.

    idx = [item]                                # Init. list with the item num.

    # psize is partial size. Poisitions in a single row.
    psize = size // shape[0]

    for i in range(ndims - 1):

        idx = idx + [idx[i] % psize]            # Add remainder to the list.
        idx[i] = idx[i] // psize                # Get pos. in the current dim.

        psize //= shape[i + 1]                  # Reduce partial size.

    # End for.

    return tuple(idx)


def strtype(x):
    """
    PURPOSE:  Convert the type of input into a short string.
    """

    return str(type(x))[8:-2]
