import sys
sys.path.append('../')

from numpy import pi
from multiZ.mdual import *

x = pi/6; h = 1.
z = x + h*eps(1) + h*eps(2)
                                        # The outputs of this
f = sin(z)                              # code are:

print('f(x)  =',  f.imag(0)         )   # f(x)  =  0.49999999999999994
print('f\'(x) =', f.imag(1)/h       )   # f'(x) =  0.8660254037844387
print('f\"(x) =', f.imag([1,2])/h**2)   # f"(x) = -0.49999999999999994
