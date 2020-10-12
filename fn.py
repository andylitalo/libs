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


def format_float(i):
    """Formats string representation of float using "-" as decimal point."""
    result = 0
    if '-' in i:
        val, dec = i.split('-')
        result = int(val) + int(dec)/10.0**(len(dec))
    else:
        result = int(i)

    return result


def get_fps(vid_filepath, prefix):
    """
    Gets the frames per second from the filepath of the video.

    Parameters
    ----------
    vid_filepath : string
        Filepath to video with frames per second in characters following prefix.
    prefix : string
        prefix given to all videos before their specs, e.g., 'v360_co2_'
    Returns
    -------
    fps : int
        frames per second of video.

    """
    i0 = vid_filepath.rfind('\\')
    filename = vid_filepath[i0:]
    i1 = filename.find(prefix) + len(prefix)
    i2 = filename[i1:].find('_')
    fps = int(filename[i1:i1+i2])
    
    return fps
    
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


def parse_vid_filepath(vid_filepath):
    i_start = vid_filepath.rfind('\\')
    vid_file = vid_filepath[i_start+1:]
    # cuts out extension and splits by underscores
    tokens = vid_file[:-4].split('_')
    prefix = ''

    for i, token in enumerate(tokens):
        if token.isnumeric():
            break
        prefix.join(token)

    params = {'prefix' : prefix,
              'fps' : int(tokens[i]),
              'exp_time' : int(tokens[i+1]),
              'Qi' : format_float(tokens[i+2]),
              'Qo' : int(tokens[i+3]),
              'd' : int(tokens[i+4]),
              'P' : int(tokens[i+5]),
              'mag' : int(tokens[i+6]),
              'num' : int(tokens[i+7])}

    return params