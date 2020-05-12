# -*- coding: utf-8 -*-
"""
Created on Thu May  7 08:33:41 2020

@author: Andy
fn.py contains useful, short, and often-used functions.
"""

import cv2
import numpy as np


def bool_2_uint8(bool_arr):
    """
    Converts a boolean array to a black-and-white
    uint8 array (0 and 255).
    PARAMETERS:
        im : (M x N) numpy array of bools
            boolean array to convert
    RETURNS:
        (result) : (M x N) numpy array of uint8s
            uint8 array of 0s (False) and 255s (True)
    """
    assert (bool_arr.dtype == 'bool'), \
        'improc.bool_2_uint8() only accepts boolean arrays.'
    return (255*bool_arr).astype('uint8')


def is_cv3():
    """
    Checks if the version of OpenCV is cv3.
    """
    (major, minor, _) = cv2.__version__.split('.')
    return int(major) == 3


def one_2_uint8(one_arr):
    """
    Converts an array of floats scaled from 0-to-1 to a
    uint8 array (0 and 255).
    PARAMETERS:
        im : (M x N) numpy array of floats
            floats from 0 to 1 to convert
    RETURNS:
        (result) : (M x N) numpy array of uint8s
            uint8 array from 0 to 255
    """
    assert (one_arr.dtype == 'float' and np.max(one_arr <= 1.0)), \
        'improc.one_2_uint8() only accepts floats arrays from 0 to 1.'
    return (255*one_arr).astype('uint8')