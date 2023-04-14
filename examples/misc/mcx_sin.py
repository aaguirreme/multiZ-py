

from numpy import pi
from multiZ.mcomplex import *

x = pi/6; h = 1e-10
z = x + h*im(1) + h*im(2)
to_cr(z)
                                        # The outputs of this
f = sin(z)                              # code are:

print('f(x)  =',  f.imag(0)         )   # f(x)  =  0.5
print('f\'(x) =', f.imag(1)/h       )   # f'(x) =  0.866025403784
print('f\"(x) =', f.imag([1,2])/h**2)   # f"(x) = -0.5
