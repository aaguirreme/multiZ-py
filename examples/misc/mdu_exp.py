import sys
sys.path.append('../')

from numpy import pi
from multiZ.mdual import *

x = np.log(2.); h = 1.
z = x + h*eps(1) + h*eps(2)

f = exp(z)

print('f(x)  =',  f.imag(0)         )   # f(x)  =  2.0
print('f\'(x) =', f.imag(1)/h       )   # f'(x) =  2.0
print('f\"(x) =', f.imag([1,2])/h**2)   # f"(x) =  2.0
