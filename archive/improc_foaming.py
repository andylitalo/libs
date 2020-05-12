# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:14:09 2019

Contains functions that are useful for image processing.

@author: Andy
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn.cluster
import skimage.measure
import cv2


def average_rgb(im):
    """
    Given an RGB image, averages RGB to produce b-w image.
    Note that the order of RGB is not important for this.
    
    Parameters:
        im : (M x N x 3) numpy array of floats or ints
            RGB image to average
            
    Returns:
        im_rgb_avg : (M x N) numpy array of floats or ints
            Black-and-white image of averaged RGB
    """
    # ensures that the image is indeed in suitable RGB format
    assert im.shape[2] == 3, "Image must be RGB (M x N x 3 array)."
    # computes average over RGB values
    res = np.round(np.mean(im, 2))
    # converts to uint8
    return res.astype('uint8')


def count_frames(path, override=False):
    """
    This method comes from https://www.pyimagesearch.com/2017/01/09/
    count-the-total-number-of-frames-in-a-video-with-opencv-and-python/
    written by Adrian Rosebrock.
    The method counts the number of frames in a video using cv2 and 
    is robust to the errors that may be encountered based on what
    dependencies the user has installed.
    
    Parameters: 
        path : string
            Direction to file of video whose frames we want to count
        override : bool (default = False)
            Uses slower, manual counting if set to True
            
    Returns:
        n_frames : int
            Number of frames in the video. -1 is passed if fails completely.
    """
    video = cv2.VideoCapture(path)
    n_frames = 0
    if override:
        n_frames = count_frames_manual(video)
    else:
        try:
            if is_cv3():
                n_frames = int(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
            else:
                n_frames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        except:
            n_frames = count_frames_manual(video)
            
    # release the video file pointer
    video.release()
    
    return n_frames


def count_frames_manual(video):
    """
    This method comes from https://www.pyimagesearch.com/2017/01/09/
    count-the-total-number-of-frames-in-a-video-with-opencv-and-python/
    written by Adrian Rosebrock.
    Counts frames in video by looping through each frame. 
    Much slower than reading the codec, but also more reliable.
    """
    # initialize the total number of frames read
    total = 0
    # loop over the frames of the video
    while True:
        # grab the current frame
        (grabbed, frame) = video.read()
        # check if we reached end of video
        if not grabbed:
            break
        # increment the total number of frames read
        total += 1
    # return the total number of frames in the video file
    return total
    
def get_angle_correction(im_labeled):
    """
    """
    rows, cols = np.where(im_labeled)
    # upper left (row, col)
    ul = (np.min(rows[cols==np.min(cols)]), np.min(cols))
    # upper right (row, col)
    ur = (np.min(rows[cols==np.max(cols)]), np.max(cols))
    # bottom left (row, col)
    bl = (np.max(rows[cols==np.min(cols)]), np.min(cols))
    # bottom right (row, col)
    br = (np.max(rows[cols==np.max(cols)]), np.max(cols))
    # angle along upper part of stream
    th_u = np.arctan((ul[0]-ur[0])/(ul[1]-ur[1]))
    # angle along lower part of stream
    th_b = np.arctan((bl[0]-br[0])/(bl[1]-br[1]))
    # compute correction by taking cosine of mean offset angle
    angle_correction = np.cos(np.mean(np.array([th_u, th_b])))

    return angle_correction


def is_cv3():
    """
    Checks if the version of OpenCV is cv3.
    """
    (major, minor, _) = cv2.__version__.split('.')
    return int(major) == 3


def measure_stream_width(im, params):
    """
    Computes the width of a stream of a darker color inside the image.
    
    Parameters:
        
    Returns:
        
    """
    # Extract parameters: microns per pixel conversion and brightfield image
    um_per_pix = params[0]
    bf = params[1]
    # Scale by brightfield image if provided
    if bf is not None:
        scale_by_brightfield(im, bf)
    # create 0-255 uint8 copy of image
    im = one_2_uint8(im)
    # K-means clustering into bkgd and stream (reshape im as array of RGB vals)
    # TODO: possible extension - group using (row,col) as well
    k_means = sklearn.cluster.KMeans(n_clusters=2).fit(im.reshape(-1,3)) 
    im_clustered = k_means.labels_.reshape(im.shape[0], im.shape[1])
    # make sure that the stream is labeled as 1 and background as 0 by using
    # the most common label for the top line as the background label
    im_clustered != (np.mean(im_clustered[0,:])>0.5)
    # Extract the longest labeled region
    im_labeled = skimage.measure.label(im_clustered)
    width_max = 0
    label = -1
    for region in skimage.measure.regionprops(im_labeled):
        row_min, col_min, row_max, col_max = region.bbox
        width = col_max - col_min
        if width > width_max:
            label = region.label
            width_max = width
    assert label >= 0, "'label'=-1', no labeled regions found in k-means clustering."
    im_stream = (im_labeled==label)
    # compute stream width and standard deviation 
    width, width_std = measure_labeled_im_width(im_stream, um_per_pix)
    
    return width, width_std
    
    
def measure_labeled_im_width(im_labeled, um_per_pix):
    """
    """
    # Count labeled pixels in each column (roughly stream width)
    num_labeled_pixels = np.sum(im_labeled, axis=0)
    # Use coordinates of corners of labeled region to correct for oblique angle
    # (assume flat sides) and convert from pixels to um
    angle_correction = get_angle_correction(im_labeled)
    stream_width_arr = num_labeled_pixels*um_per_pix*angle_correction
    # remove columns without any stream (e.g., in case of masking)
    stream_width_arr = stream_width_arr[stream_width_arr > 0]
    # get mean and standard deviation
    mean = np.mean(stream_width_arr)
    std = np.std(stream_width_arr)
    
    return mean, std

    
def one_2_uint8(im, copy=True):
    """
    Returns a copy of an image scaled from 0 to 1 as an image of uint8's scaled
    from 0-255.
    
    Parameters:
        im : 2D or 3D array of floats
            Image with pixel intensities measured from 0 to 1 as floats
        copy : bool, optional
            If True, image will be copied first
    
    Returns:
        im_uint8 : 2D or 3D array of uint8's
            Image with pixel intensities measured from 0 to 255 as uint8's
    """
    if copy:
        im = np.copy(im)
    return (255*im).astype('uint8')

def proc_im_seq(im_path_list, proc_fn, params, columns=None):
    """
    Processes a sequence of images with the given function and returns the
    results. Images are provided as filepaths to images, which are loaded (and
    possibly copied before analysis to preserve the image).
    
    Parameters:
        im_path_list : array-like
            Sequence of filepaths to the images to be processed
        proc_fn : function handle
            Handle of function to use to process image sequence
        params : list
            List of parameters to plug into the processing function
        columns : array-like, optional
            Results from image processing are saved to this dataframe
    
    Returns:
        output : list (optionally, Pandas DataFrame if columns given)
            Results from image processing        
    """
    # Initialize list to store results from image processing
    output = []
    # Process each image in sequence.
    for im_path in im_path_list:
        print("Begin processing {im_path}.".format(im_path=im_path))
        im = plt.imread(im_path)
        output += [proc_fn(im, params)]
    # If columns provided, convert list into a dataframe
    if columns:
        output = pd.DataFrame(output, columns=columns)      
        
    return output


def scale_by_brightfield(im, bf):
    """
    scale pixels by value in brightfield
    
    Parameters:
        im : array of floats or ints
            Image to scale
        bf : array of floats or ints
            Brightfield image for scaling given image
    
    Returns:
        im_scaled : array of uint8
            Image scaled by brightfield image
    """
    # convert to intensity map scaled to 1.0
    bf_1 = bf.astype(float) / 255.0
    # scale by bright field
    im_scaled = np.divide(im,bf_1)
    # rescale result to have a max pixel value of 255
    im_scaled *= 255.0/np.max(im_scaled)
    # change type to uint8
    im_scaled = im_scaled.astype('uint8')

    return im_scaled