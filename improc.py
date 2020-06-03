"""
improc.py contains definitions of methods for image-processing
using OpenCV with displays in Bokeh.
"""

# imports standard libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import time

# imports bokeh modules
from bokeh.io import show, push_notebook

# imports image-processing-specific libraries 
import skimage.morphology
import sklearn.cluster
import skimage.measure
import skimage.color
import skimage.segmentation
import skimage.feature
import skimage.filters
import cv2
import scipy.ndimage

# imports custom libraries
import vid
import fn
import plotimproc as plot

# imports custom classes (with reload clauses)
from importlib import reload
import classes
reload(classes)
from classes import Bubble


def adjust_brightness(im, brightness, sat=255):
    """
    Adjusts brightness of image by scaling all pixels by the 
    `brightness` parameter.
    
    Parameters
    ----------
    im : (M, N, P) numpy array
        Image whose brightness is scaled. P >= 3,
        so RGB, BGR, RGBA acceptable.
    brightness : float
        Scaling factor for pixel values. If < 0, returns junk.
    sat : int
        Saturation value for pixels (usually 255 for uint8)
    
    Returns
    -------
    im : (M, N, P) numpy array
        Original image with pixel values scaled          
        
    """
    # if image is 4-channel (e.g., RGBA) extracts first 3
    is_rgba = (len(im.shape) == 3) and (im.shape[2] == 4)
    if is_rgba:
        im_to_scale = im[:,:,:3]
    else:
        im_to_scale = im
    # scales pixel values in image
    im_to_scale = im_to_scale.astype(float)
    im_to_scale *= brightness
    # sets oversaturated values to saturation value
    im_to_scale[im_to_scale >= sat] = sat
    # loads result into original image
    if is_rgba:
        im[:,:,:3] = im_to_scale.astype('uint8')
    else:
        im = im_to_scale.astype('uint8')
    
    return im


def average_rgb(im):
    """
    Given an RGB image, averages RGB to produce b-w image.
    Note that the order of RGB is not important for this.
    
    Parameters
    ----------
    im : (M x N x 3) numpy array of floats or ints
        RGB image to average
            
    Returns
    -------
    im_rgb_avg : (M x N) numpy array of floats or ints
        Black-and-white image of averaged RGB
    
    """
    # ensures that the image is indeed in suitable RGB format
    assert im.shape[2] == 3, "Image must be RGB (M x N x 3 array)."
    # computes average over RGB values
    res = np.round(np.mean(im, 2))
    # converts to uint8
    return res.astype('uint8')


def assign_bubbles(frame_labeled, f, bubbles_prev, bubbles_archive, ID_curr, 
                   flow_dir, fps, pix_per_um, width_border):
    """
    Assigns Bubble objects with unique IDs to the labeled objects on the video
    frame provided. This method is used on a single frame in the context of
    processing an entire video and uses information from the previous frame.
    Inspired by and partially copied from PyImageSearch [1].
    
    Updates bubbles_prev and bubbles_archive in place.
    
    Parameters
    ----------
    frame_labeled : (M x N) numpy array of uint8
        Video frame with objects labeled with unique numbers. They only need
        to be unique so regionprops() can distinguish them. These are not
        the values used to determine unique ID numbers for the bubbles.
    f : int
        Frame number from video
    bubbles_prev : OrderedDict of dictionaries
        Ordered dictionary of dictionaries of properties of bubbles 
        from the previous frame
    bubbles_archive : OrderedDict of Bubble objects
        Ordered dictionary of bubbles from all previous frames
    ID_curr : int
        Next ID number to be assigned (increasing order)
    flow_dir : numpy array of 2 floats
        Unit vector indicating the flow direction
    fps : float
        Frames per second of video
    pix_per_um : float
        Conversion of pixels per micron (um)
    width_border : int
        Number of pixels to remove from border for image processign in
        highlight_bubble()
        
    Returns
    -------
    ID_curr : int
        Updated value of next ID number to assign.        
     
    References
    ----------
    .. [1] https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
    """
    
    # identifies the different objects in the frame
    region_props = skimage.measure.regionprops(frame_labeled)
    # creates dictionaries of properties for each object
    bubbles_curr = []
    for i, props in enumerate(region_props):
        # creates dictionary of bubble properties for one frame, which
        # can be merged to a Bubble object
        bubble = {}
        bubble['centroid'] = props.centroid
        bubble['area'] = props.area
        bubble['frame'] = f
        bubble['on border'] = is_on_border(props.bbox, 
              frame_labeled, width_border)
        # adds dictionary for this bubble to list of bubbles in current frame
        bubbles_curr += [bubble]
        
    # if no bubbles seen in previous frame, assigns bubbles in current frame
    # to new bubble IDs 
    if len(bubbles_prev) == 0:
        for i in range(len(bubbles_curr)):
            bubbles_prev[ID_curr] = bubbles_curr[i]
            ID_curr += 1
   
    # if no bubbles in current frame, removes bubbles from dictionary of
    # bubbles in the previous frame
    elif len(bubbles_curr) == 0:
        for ID in bubbles_prev.keys():
            del bubbles_prev[ID]         
   
    # otherwise, assigns bubbles in current frames to previous objects based
    # on distance off flow axis and other parameters (see bubble_distance())
    else:
        # grabs the set of object IDs from the previous farme
        IDs = list(bubbles_prev.keys())
        # computes M x N matrix of distances (M = # bubbles in previous frame,
        # N = # bubbles in current frame)
        d_mat = bubble_d_mat(list(bubbles_prev.values()), 
                             bubbles_curr, flow_dir)
        
        ### SOURCE: Much of the next is directly copied from [1]
        # in order to perform this matching we must (1) find the
        # smallest value in each row and then (2) sort the row
        # indexes based on their minimum values so that the row
        # with the smallest value is at the *front* of the index
        # list
        rows = d_mat.min(axis=1).argsort()
        # next, we perform a similar process on the columns by
        # finding the smallest value in each column and then
        # sorting using the previously computed row index list
        cols = d_mat.argmin(axis=1)[rows]
        # in order to determine if we need to update, register,
        # or deregister an object we need to keep track of which
        # of the rows and column indexes we have already examined
        rows_used = set()
        cols_used = set()
        
        # loops over the combination of the (row, column) index
        # tuples
        for (row, col) in zip(rows, cols):
            # if we have already examined either the row or
            # column value before, ignores it
            if row in rows_used or col in cols_used:
                continue
            # also ignores pairings where the second bubble is upstream
            # these pairings are marked with a penalty that is 
            # larger than the largest distance across the frame
            d_longest = np.linalg.norm(frame_labeled.shape)
            if d_mat[row, col] > d_longest:
                continue
            
            # otherwise, grabs the object ID for the current row,
            # set its new centroid, and reset the disappeared
            # counter
            ID = IDs[row]
            bubbles_prev[ID] = bubbles_curr[col]
            # indicates that we have examined each of the row and
            # column indexes, respectively
            rows_used.add(row)
            cols_used.add(col)
            
        # computes both the row and column index we have NOT yet
        # examined
        rows_unused = set(range(0, d_mat.shape[0])).difference(rows_used)
        cols_unused = set(range(0, d_mat.shape[1])).difference(cols_used)
        
        # loops over the unused row indexes to remove bubbles that disappeared
        for row in rows_unused:
            # grabs the object ID for the corresponding row
            # index and save to archive
            ID = IDs[row]
            # removes bubble from dictionary for previous frame
            del bubbles_prev[ID]

        # registers each unregistered new input centroid as a bubble seen
        for col in cols_unused:
            bubbles_prev[ID_curr] = bubbles_curr[col]
            ID_curr += 1
     
    # archives bubbles from this frame in order of increasing ID
    for ID in bubbles_prev.keys():
        # creates new ordered dictionary of bubbles if new bubble
        if ID == len(bubbles_archive):
            bubbles_archive[ID] = Bubble(ID, fps, frame_labeled.shape, flow_dir, 
                           pix_per_um, props_raw=bubbles_prev[ID])
        elif ID < len(bubbles_archive):
            bubbles_archive[ID].add_props(bubbles_prev[ID])
        else:
            print('In assign_bubbles(), IDs looped out of order while saving to archive.')
                
    return ID_curr


def bubble_distance(bubble1, bubble2, axis, upstream_penalty=1E10):
    """
    Computes the distance between each pair of points in the two sets
    perpendicular to the axis. All inputs must be numpy arrays.
    # TODO incorporate velocity profile into objective
    """
    c1 = np.array(bubble1['centroid'])
    c2 = np.array(bubble2['centroid'])
    diff = c2 - c1
    # computes component of projection along axis
    comp = np.dot(diff, axis)
    # computes projection along flow axis
    proj =  comp*axis
    d_off_axis = np.linalg.norm(diff - proj)
    # adds huge penalty if second bubble is upstream of first bubble
    d = d_off_axis + upstream_penalty*(comp < 0)
    
    return d


def bubble_d_mat(bubbles1, bubbles2, axis):
    """
    Computes the distance between each pair of bubbles and organizes into a 
    matrix.
    """
    M = len(bubbles1)
    N = len(bubbles2)
    d_mat = np.zeros([M, N])
    for i, bubble1 in enumerate(bubbles1):
        for j, bubble2 in enumerate(bubbles2):
            d_mat[i,j] = bubble_distance(bubble1, bubble2, axis)
            
    return d_mat
   

def get_angle_correction(im_labeled):
    """
    correction of length due to offset angle of stream
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


def get_frame_IDs(bubbles_archive, start_frame, end_frame, skip):
    """Returns list of IDs of bubbles in each frame"""
    # initializes dictionary of IDs for each frame
    frame_IDs = {}
    for f in range(start_frame, end_frame, skip):
        frame_IDs[f] = []
    # loads IDs of bubbles found in each frame 
    for ID in bubbles_archive.keys():
        bubble = bubbles_archive[ID]
        frames = bubble.get_props('frame')
        for f in frames:
            frame_IDs[f] += [ID]
                
    return frame_IDs


def get_points(Npoints=1,im=None):
    """ Alter the built in ginput function in matplotlib.pyplot for custom use.
    This version switches the function of the left and right mouse buttons so
    that the user can pan/zoom without adding points. NOTE: the left mouse
    button still removes existing points.
    INPUT:
        Npoints = int - number of points to get from user clicks.
    OUTPUT:
        pp = list of tuples of (x,y) coordinates on image
    """
    if im is not None:
        plt.imshow(im)
        plt.axis('image')

    pp = plt.ginput(n=Npoints,mouse_add=3, mouse_pop=1, mouse_stop=2,
                    timeout=0)
    return pp


def get_val_channel(frame, selem=None):
    """
    Returns the value channel of the given frame.
    """
    # Convert reference frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Only interested in "value" channel to distinguish bubbles, filters result
    val = skimage.filters.median(hsv[:,:,2], selem=selem).astype('uint8')
    
    return val

    
def highlight_bubble(frame, ref_frame, thresh, width_border, selem, min_size,
                     ret_all_steps=False):
    """
    Highlights bubbles (regions of different brightness) with white and
    turns background black. Ignores edges of the frame.
    Only accepts 2D frames.
    """
    assert (len(frame.shape) == 2) and (len(ref_frame.shape) == 2), \
        'improc.highlight_bubble() only accepts 2D frames.'
    
    # subtracts reference image from current image (value channel)
    im_diff = cv2.absdiff(ref_frame, frame)
    # thresholds image to become black-and-white
    thresh_bw = thresh_im(im_diff, thresh)
    # smooths out thresholded image
    closed_bw = skimage.morphology.binary_closing(thresh_bw, selem=selem)
    # removes small objects
    bubble_bw = skimage.morphology.remove_small_objects(closed_bw.astype(bool),
                                                        min_size=min_size)
    # converts image to uint8 type from bool
    bubble_bw = 255*bubble_bw.astype('uint8')
    # fills enclosed holes with white, but leaves open holes black
    bubble_part_filled = scipy.ndimage.morphology.binary_fill_holes(bubble_bw)    
    # removes any white pixels slightly inside the border of the image
    mask_full = np.ones([bubble_part_filled.shape[0]-2*width_border, 
                        bubble_part_filled.shape[1]-2*width_border])
    mask_deframe = np.zeros_like(bubble_part_filled)
    mask_deframe[width_border:-width_border,width_border:-width_border] = mask_full
    mask_frame_sides = np.logical_not(mask_deframe)
    # make the tops and bottoms black so only the sides are kept
    mask_frame_sides[0:width_border,:] = 0
    mask_frame_sides[-width_border:-1,:] = 0
    # frames sides of filled bubble image
    bubble_framed = np.logical_or(bubble_part_filled, mask_frame_sides)
    # fills in open space in the middle of the bubble that might not get
    # filled if bubble is on the edge of the frame (because then that
    # open space is not completely bounded)
    bubble_filled = scipy.ndimage.morphology.binary_fill_holes(bubble_framed)
    bubble = mask_im(bubble_filled, np.logical_not(mask_frame_sides))
    bubble = 255*bubble.astype('uint8')
    
    # returns intermediate steps if requeseted.
    if ret_all_steps:
        return im_diff, thresh_bw, closed_bw, bubble_bw, \
                bubble_part_filled, bubble
    else:
        return bubble
    
    
def highlight_bubble_test(frame, ref_frame, th_lo, th_hi, width_border, selem,
                          min_size, ret_all_steps=False):
    """
    Version of highlight_bubble() for testing new features. This was 
    created so code that uses the original version can still run the same.
    
    Highlights bubbles (regions of different brightness) with white and
    turns background black. Ignores edges of the frame.
    Only accepts 2D frames.
    """
    assert (len(frame.shape) == 2) and (len(ref_frame.shape) == 2), \
        'improc.highlight_bubble() only accepts 2D frames.'
    assert th_lo < th_hi, \
        'In improc.highlight_bubbles_test(), low threshold must be lower.'
    
    # subtracts reference image from current image (value channel)
    im_diff = cv2.absdiff(ref_frame, frame)
    # thresholds image to become black-and-white
    thresh_bw = skimage.filters.apply_hysteresis_threshold(\
                        im_diff, th_lo, th_hi)


    # smooths out thresholded image
    closed_bw = skimage.morphology.binary_closing(thresh_bw, selem=selem)
    # removes small objects
    bubble_bw = skimage.morphology.remove_small_objects(closed_bw.astype(bool),
                                                        min_size=min_size)
    # converts image to uint8 type from bool
    bubble_bw = 255*bubble_bw.astype('uint8')
    # fills enclosed holes with white, but leaves open holes black
    bubble_part_filled = scipy.ndimage.morphology.binary_fill_holes(bubble_bw)    
    # removes any white pixels slightly inside the border of the image
    mask_full = np.ones([bubble_part_filled.shape[0]-2*width_border, 
                        bubble_part_filled.shape[1]-2*width_border])
    mask_deframe = np.zeros_like(bubble_part_filled)
    mask_deframe[width_border:-width_border,width_border:-width_border] = mask_full
    mask_frame_sides = np.logical_not(mask_deframe)
    # make the tops and bottoms black so only the sides are kept
    mask_frame_sides[0:width_border,:] = 0
    mask_frame_sides[-width_border:-1,:] = 0
    # frames sides of filled bubble image
    bubble_framed = np.logical_or(bubble_part_filled, mask_frame_sides)
    # fills in open space in the middle of the bubble that might not get
    # filled if bubble is on the edge of the frame (because then that
    # open space is not completely bounded)
    bubble_filled = scipy.ndimage.morphology.binary_fill_holes(bubble_framed)
    bubble = mask_im(bubble_filled, np.logical_not(mask_frame_sides))
    bubble = 255*bubble.astype('uint8')
    
    # returns intermediate steps if requeseted.
    if ret_all_steps:
        return im_diff, thresh_bw, closed_bw, bubble_bw, \
                bubble_part_filled, bubble
    else:
        return bubble


def is_on_border(bbox, im, width_border):
    """
    Checks if object is on the border of the frame.
    
    bbox is (min_row, min_col, max_row, max_col). Pixels belonging to the 
    bounding box are in the half-open interval [min_row; max_row) and 
    [min_col; max_col).
    """
    min_col = bbox[1]
    max_col = bbox[3] 
    width_im = im.shape[1]
    if (min_col <= width_border) or (max_col >= width_im - width_border):
        return True
    else:
        return False


def mask_im(im, mask):
    """
    Returns image with all pixels outside mask blacked out
    mask is boolean array or array of 0s and 1s of same shape as image
    """
    # Applies mask depending on dimensions of image
    tmp = np.shape(im)
    im_masked = np.zeros_like(im)
    if len(tmp) == 3:
        for i in range(3):
            im_masked[:,:,i] = mask*im[:,:,i]
    else:
        im_masked = im*mask

    return im_masked


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


def rotate_image(im,angle,center=[],crop=False,size=None):
    """
    Rotate the image about the center of the image or the user specified
    center. Rotate by the angle in degrees and scale as specified. The new
    image will be square with a side equal to twice the length from the center
    to the farthest.
    """
    temp = im.shape
    height = temp[0]
    width = temp[1]
    # Provide guess for center if none given (use midpoint of image window)
    if len(center) == 0:
        center = (width/2.0,height/2.0)
    if not size:
        tempx = max([height-center[1],center[1]])
        tempy = max([width-center[0],center[0]])
        # Calculate dimensions of resulting image to contain entire rotated image
        L = int(2.0*np.sqrt(tempx**2.0 + tempy**2.0))
        midX = L/2.0
        midY = L/2.0
        size = (L,L)
    else:
        midX = size[1]/2.0
        midY = size[0]/2.0

    # Calculation translation matrix so image is centered in output image
    dx = midX - center[0]
    dy = midY - center[1]
    M_translate = np.float32([[1,0,dx],[0,1,dy]])
    # Calculate rotation matrix
    M_rotate = cv2.getRotationMatrix2D((midX,midY),angle,1)
    # Translate and rotate image
    im = cv2.warpAffine(im,M_translate,(size[1],size[0]))
    im = cv2.warpAffine(im,M_rotate,(size[1],size[0]),flags=cv2.INTER_LINEAR)
    # Crop image
    if crop:
        (x,y) = np.where(im>0)
        im = im[min(x):max(x),min(y):max(y)]

    return im


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


def scale_image(im,scale):
    """
    Scale the image by multiplicative scale factor "scale".
    """
    temp = im.shape
    im = cv2.resize(im,(int(scale*temp[1]),int(scale*temp[0])))

    return im


def thresh_im(im, thresh=-1, c=5):
    """
        Applies a threshold to the image and returns black-and-white result.
    c : int 
        channel to threshold
    Modified from ImageProcessingFunctions.py
    """
    n_dims = len(im.shape)
    if n_dims == 3:
        if c > n_dims:
            c_brightest = -1
            pix_val_brightest = -1
            for i in range(n_dims):
                curr_brightest = np.max(im[:,:,i])
                if curr_brightest > pix_val_brightest:
                    c_brightest = i
                    pix_val_brightest = curr_brightest
        else:
            c_brightest = c
        im = np.copy(im[:,:,c_brightest])
    if thresh == -1:
        # Otsu's thresholding
        ret, thresh_im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        ret, thresh_im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)

    return thresh_im


def track_bubble(vid_filepath, n_ref, thresh, flow_dir, pix_per_um,
                 width_border=10, selem=None, ret_IDs=False,
                 min_size=None, start_frame=0, end_frame=-1, skip=1):
    """
    """
    # gets value channel of reference frame from video
    ref_frame, _ = vid.load_frame(vid_filepath, n_ref, bokeh=False)
    ref_val = get_val_channel(ref_frame, selem=selem) 
    # gets number of frames from video
    fps = vid.count_frames(vid_filepath)
    # initializes ordered dictionary of bubble data from past frames and archive of all data
    bubbles_prev = OrderedDict()
    bubbles_archive = OrderedDict()
    # initializes counter of current bubble label (0-indexed)
    ID_curr = 0    
    # chooses end frame to be last frame if given as -1
    if end_frame == -1:
        end_frame = vid.count_frames(vid_filepath)
        
    # loops through frames of video
    for f in range(start_frame, end_frame, skip):
        frame, _ = vid.load_frame(vid_filepath, f, bokeh=False)
        val = get_val_channel(frame, selem=selem)
        # highlights bubbles in the given frame
        bubbles_bw = highlight_bubble(val, ref_val, thresh, width_border, selem, min_size)
        # finds bubbles and assigns IDs to track them, saving to archive
        frame_labeled = skimage.measure.label(bubbles_bw)
        ID_curr = assign_bubbles(frame_labeled, f, bubbles_prev, bubbles_archive,
                                 ID_curr, flow_dir, fps, pix_per_um, width_border)

    # only returns IDs for each frame if requested
    if ret_IDs:
        frame_IDs = get_frame_IDs(bubbles_archive, start_frame, end_frame, skip)
        return bubbles_archive, frame_IDs
    else:
        return bubbles_archive

    
def test_track_bubble(vid_filepath, bubbles, frame_IDs, n_ref, thresh, pix_per_um, 
                      width_border=10, selem=None, min_size=None, start_frame=0, 
                      end_frame=-1, skip=1, num_colors=10, time_sleep=2, 
                      brightness=3.0, fig_size_red=0.5):
    """
    """
#    # creates colormap
#    cmap = cm.get_cmap('Spectral')

    # gets value channel of reference frame from video
    ref_frame, _ = vid.load_frame(vid_filepath, n_ref, bokeh=False)
    ref_val = get_val_channel(ref_frame, selem=selem) 
    # chooses end frame to be last frame if given as -1
    if end_frame == -1:
        end_frame = vid.count_frames(vid_filepath)
        
    # creates figure for displaying frames with labeled bubbles
    p, im = plot.format_frame(plot.bokehfy(ref_val), pix_per_um, fig_size_red, brightness=brightness)
    show(p, notebook_handle=True)
    # loops through frames
    for f in range(start_frame, end_frame, skip):   
        # gets value channel of frame for processing
        frame, _ = vid.load_frame(vid_filepath, f, bokeh=False)
        val = get_val_channel(frame, selem=selem) 
        # processes frame
        bubble = highlight_bubble(val, ref_val, thresh, width_border, selem, min_size)
        # labels bubbles
        frame_labeled, num_labels = skimage.measure.label(bubble, return_num=True)
        # ensures that the same number of IDs are provided as objects found
        assert len(frame_IDs[f]) == num_labels, \
            'improc.test_track_bubble() has label mismatch for frame {0:d}.'.format(f) \
            + 'num_labels = {0:d} and number of frame_IDs = {1:d}'.format(num_labels, len(frame_IDs[f]))
        # assigns IDs to pixels of each bubble--sort of helps with color-coding
        IDs = frame_IDs[f]
        frame_relabeled = np.zeros(frame_labeled.shape)
        for ID in IDs:
            # finds label associated with the bubble with this id
            rc, cc = bubbles[ID].get_prop('centroid', f)
            label = frame_labeled[int(rc), int(cc)]
            frame_relabeled[frame_labeled==label] = (ID+1)%255 # re-indexes from 1
        frame_colored = skimage.color.label2rgb(frame_relabeled, image=frame, 
                                                bg_label=0)
 
        # converts frame from one-scaled to 255-scaled
        frame_disp = fn.one_2_uint8(frame_colored)
        for ID in IDs:
            centroid = bubbles[ID].get_prop('centroid', f)
            x = int(centroid[1])
            y = int(centroid[0])
            white = (255, 255, 255)
            black = (0, 0, 0)
            if bubbles[ID].get_prop('on border', f):
                color = black
            else:
                color = white
            frame_disp = cv2.putText(img=frame_disp, text=str(ID), org=(x, y), 
                                    fontFace=0, fontScale=2, color=color, 
                                    thickness=3)
        frame_disp = adjust_brightness(plot.bokehfy(frame_disp), brightness)
        im.data_source.data['image']=[frame_disp]
        push_notebook()
        time.sleep(time_sleep)
        
    return p