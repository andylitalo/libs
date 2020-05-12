"""
improc.py contains definitions of methods for image-processing
using OpenCV with displays in Bokeh.
"""

# imports standard libraries
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, push_notebook
from bokeh.models import Range1d, Label
from bokeh.models.annotations import Title
from bokeh.layouts import gridplot
import time

# imports image-processing-specific libraries 
import skimage.morphology
import skimage.measure
import skimage.color
import skimage.segmentation
import skimage.feature
import cv2
from scipy import ndimage

# imports custom libraries
import UserInputFunctions as UIF

def adjust_brightness(frame, brightness):
    """
    """
    rgb = frame[:,:,:3]
    rgb = rgb.astype(float)
    rgb *= brightness
    rgb[rgb >= 255] = 255
    frame[:,:,:3] = rgb.astype('uint8')
    
    return frame


def bokehfy(frame, vert_flip=0):
    """
    """
    # converts boolean images to uint8
    if frame.dtype == 'bool':
        frame = 255*frame.astype('uint8')
    # converts BGR images to RGBA (from CV2, which uses BGR instead of RGB)
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # because Bokeh expects a RGBA image
    # converts gray scale (2d) images to RGBA
    elif len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
    frame = cv2.flip(frame, vert_flip) # because Bokeh flips vertically
    
    return frame


def bool_2_uint8(bool_arr):
    """Converts boolean array to uint8, scaled from 0 to 255."""
    assert (bool_arr.dtype == 'bool'), 'improc.bool_2_uint8() only accepts boolean arrays.'
    return (255*bool_arr).astype('uint8')


def click_flow_dir(im):
    """
    User clicks along the inner wall of the capillary to indicate the flow
    direction in a vector (tuple). User must enter the inline command
    "%matplotlib qt" to get a pop-out plot for using this function.
    """
    # formats image for use in matplotlib's imshow
    im_p = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_p = 255.0 / np.max(im_p) * im_p
    im_p = im_p.astype('uint8')
    # collects two points from clicks defining the flow direction from the inner wall
    xy_vals = UIF.define_outer_edge(im_p, 'polygon', 'right-click 2 pts left-to-right along inner wall, then center-click')
    # computes coordinates of vector from clicks
    dx = xy_vals[1][0]-xy_vals[0][0]
    dy = xy_vals[1][1]-xy_vals[0][1]
    # normalizes flow direction
    d = np.sqrt(dx**2 + dy**2)
    flow_dir = (dx / d, dy / d)
 
    return flow_dir
               
    
def find_bubble(frame, ref_frame, thresh, width_frame, selem, min_size):
    """
    Highlights bubbles (regions of different brightness) with white and
    turns background black. Ignores edges of the frame.
    Only accepts 2D frames.
    """
    assert (len(frame.shape) == 2) and (len(ref_frame.shape) == 2), 'improc.find_bubble() only accepts 2D frames.'
    
    # subtracts reference image from current image (value channel)
    im_diff = cv2.absdiff(ref_frame, frame)
    # thresholds image to become black-and-white
    thresh_bw = thresh_im(im_diff, thresh)
    # smooths out thresholded image
    closed_bw = skimage.morphology.binary_closing(thresh_bw, selem=selem)
    # removes small objects
    bubble_bw = skimage.morphology.remove_small_objects(closed_bw.astype(bool), min_size=min_size)
    # converts image to uint8 type from bool
    bubble_bw = 255*bubble_bw.astype('uint8')
    # fills enclosed holes with white, but leaves open holes black
    bubble_part_filled = ndimage.morphology.binary_fill_holes(bubble_bw)    
    # removes any white pixels slightly inside the border of the image
    mask_full = np.ones([bubble_part_filled.shape[0]-2*width_frame, 
                        bubble_part_filled.shape[1]-2*width_frame])
    mask_deframe = np.zeros_like(bubble_part_filled)
    mask_deframe[width_frame:-width_frame,width_frame:-width_frame] = mask_full
    mask_frame_sides = np.logical_not(mask_deframe)
    # make the tops and bottoms black so only the sides are kept
    mask_frame_sides[0:width_frame,:] = 0
    mask_frame_sides[-width_frame:-1,:] = 0
    # frames sides of filled bubble image
    bubble_framed = np.logical_or(bubble_part_filled, mask_frame_sides)
    # fills in open space in the middle of the bubble that might not get
    # filled if bubble is on the edge of the frame (because then that
    # open space is not completely bounded)
    bubble = mask_im(ndimage.morphology.binary_fill_holes(bubble_framed), np.logical_not(mask_frame_sides))
    bubble = 255*bubble.astype('uint8')
    
    return bubble

def format_frame(frame, pix_per_um, fig_size_red, brightness=1.0, title=None):
    frame = adjust_brightness(frame, brightness)
    width = frame.shape[1]
    height= frame.shape[0]
    width_um = int(width / pix_per_um)
    height_um = int(height / pix_per_um)
    width_fig = int(width*fig_size_red)
    height_fig = int(height*fig_size_red)
    p = figure(x_range=(0,width_um), y_range=(0,height_um), output_backend="webgl", width=width_fig, height=height_fig, title=title)
    p.xaxis.axis_label = 'width [um]'
    p.yaxis.axis_label = 'height [um]'
    im = p.image_rgba(image=[frame], x=0, y=0, dw=width_um, dh=height_um)
    
    return p, im
   
    
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


def linked_four_frames(four_frames, pix_per_um, fig_size_red, show_fig=True):
    """
    Shows four frames with linked panning and zooming.
    """
    # list of figures
    p = []
    # creates images
    for frame in four_frames:
        p_new, _ = format_frame(frame, pix_per_um, fig_size_red)
        p += [p_new]
    # sets ranges
    for i in range(1, len(p)):
        p[i].x_range = p[0].x_range # links horizontal panning
        p[i].y_range = p[0].y_range # links vertical panning
    # creates gridplot
    p_grid = gridplot([[p[0], p[1]], [p[2], p[3]]])
    # shows figure
    if show_fig:
        show(p_grid)
    
    return p_grid


def linked_frames(frame_list, pix_per_um, fig_size_red, shape=(2,2), show_fig=True, brightness=1.0, title_list=[]):
    """
    Shows four frames with linked panning and zooming.
    """
    # list of figures
    p = []
    # creates images
    for frame in frame_list:
        p_new, _ = format_frame(frame, pix_per_um, fig_size_red, brightness=brightness)
        p += [p_new]
    # adds titles to plots if provided (with help from
    # https://stackoverflow.com/questions/47733953/set-title-of-a-python-bokeh-plot-figure-from-outside-of-the-figure-functio)
    for i in range(len(title_list)):
        t = Title()
        t.text = title_list[i]
        p[i].title = t
    # sets ranges
    for i in range(1, len(p)):
        p[i].x_range = p[0].x_range # links horizontal panning
        p[i].y_range = p[0].y_range # links vertical panning
    # formats list for gridplot
    n = 0
    p_table = []
    for r in range(shape[0]):
        p_row = []
        for c in range(shape[1]):
            if n >= len(p):
                break
            p_row += [p[n]]
            n += 1
        p_table += [p_row]
        if n >= len(p):
            break
            
    # creates gridplot
    p_grid = gridplot(p_table)
    # shows figure
    if show_fig:
        show(p_grid)
    
    return p_grid

    
def load_frame(vid_filepath, num, vert_flip=0, bokeh=True):
    """Loads frame from video using OpenCV and prepares for display in Bokeh."""
    cap = cv2.VideoCapture(vid_filepath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, num)
    ret, frame = cap.read()
    if bokeh:
        frame = bokehfy(frame)
    
    return frame, cap


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


def show_frame(vid_filepath, start_frame, pix_per_um, vert_flip=0, fig_size_red=0.5, brightness=1.0, show_fig=True):
    """
    vert_flip:  # code for flipping vertically with cv2.flip()
    # dw is the label on the axis
    # x_range is the extent shown (so it should match dw)
    # width is the width of the figure box
    """
    frame, cap = load_frame(vid_filepath, start_frame)
    p, im = format_frame(frame, pix_per_um, fig_size_red, brightness=brightness)
    if show_fig:
        show(p, notebook_handle=True)

    return p, im, cap


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
    
    
def view_video(vid_filepath, start_frame, pix_per_um, sleep_time=0.3, brightness=1.0, vert_flip=0):
    """
    vert_flip:  # code for flipping vertically with cv2.flip()
    Functions and setup were adapted from:
    https://stackoverflow.com/questions/27882255/is-it-possible-to-display-an-opencv-video-inside-the-ipython-jupyter-notebook
    """
    p, im, cap = show_frame(vid_filepath, start_frame, pix_per_um, brightness=brightness)
    while True:
        ret, frame = cap.read()
        frame=adjust_brightness(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA), brightness)
        frame=cv2.flip(frame, vert_flip)
        im.data_source.data['image']=[frame]
        push_notebook()
        time.sleep(sleep_time)
    