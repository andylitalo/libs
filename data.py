# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:31:58 2019

Defines functions useful for loading, manipulating, and saving data and
metadata.

@author: Andy
"""

import glob
import os


def get_filepaths(path, template):
    """
    Returns a list of filepaths to files inside the given folder that start 
    with the given header.
    
    Parameters:
        path : string
            Path to folder of files of interest
        template : string
            Template for file names, using "*" for varying parts of file name
    
    Returns:
        file_list : list
            List of filepaths to the desired files
    """
    # Get file path
    filepath_structure = os.path.join(path, template)
    file_list = glob.glob(filepath_structure)
    
    return file_list


