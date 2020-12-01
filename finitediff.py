"""
@finitediff.py contains functions for performing finite difference calculations.

@author Andy Ylitalo
@date December 1, 2020
"""



def dydx_fwd_2nd(y0, y1, y2, dx):
    """
    Computes 2nd-order forward difference of dy/dx with a Taylor scheme.
    """
    dydx = (-3*y0 + 4*y1 - y2)/(2*dx)

    return dydx
