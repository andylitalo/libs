# -*- coding: utf-8 -*-
"""
Created on Thu May  7 08:55:33 2020

@author: Andy
plot.py contains basic plotting functions
to be used within the functions of the library.
"""

import matplotlib.pyplot as plt



def no_ticks(image):
    """
    This removes tick marks and numbers from the axes of the image and fills 
    up the figure window so the image is easier to see.
    """
    plt.imshow(image)
    plt.axis('off')
    plt.axis('image')
    plt.tight_layout(pad=0)
