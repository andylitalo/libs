"""finitediff.py contains functions for performing finite difference calculations.

Author : Andy Ylitalo
Date : December 1, 2020
"""



def d2ydx2_non_1st(y_arr, x_arr):
    """
    Computes 1st-order second derivative d2y/dx2 with a Taylor scheme for a
    nonuniform grid.
    Uses eqn 2.21 from Parviz Moin's "Fundamentals of Engineering Numerical
    Analysis" (p. 23)
    """
    y0 = y_arr[:-2]
    y1 = y_arr[1:-1]
    y2 = y_arr[2:]
    h1 = x_arr[1:-1] - x_arr[:-2]
    h2 = x_arr[2:] - x_arr[1:-1]

    return 2*( y0 / (h1*(h1+h2)) - y1/(h1*h2) + y2/(h2*(h1+h2)) )

def dydx_cd_2nd(y_arr, dx):
    """
    Computes 2nd-order central difference of dy/dx with a Taylor scheme.
    """
    return (y_arr[2:] - y_arr[:-2]) / (2*dx)

def dydx_non_1st(y_arr, x_arr):
    """
    Computes 1st-order central difference of dy/dx with a Taylor scheme for a
    nonuniform grid.
    Uses eqn 2.20 from Parviz Moin's "Fundamentals of Engineering Numerical
    Analysis" (p. 23)
    """
    return (y_arr[2:] - y_arr[:-2]) / (x_arr[2:] - x_arr[:-2])

def dydx_fwd_2nd(y0, y1, y2, dx):
    """
    Computes 2nd-order forward difference of dy/dx with a Taylor scheme.
    """
    return (-3*y0 + 4*y1 - y2)/(2*dx)
