# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:52:03 2015

@author: John
Collection of function definitions for use in image processing.
"""

import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import pickle as pkl
from skimage import filters
import skimage.morphology
from pandas import unique
from scipy.optimize import least_squares
from scipy.signal import medfilt

# Custom modules
import Functions as Fun
import UserInputFunctions as UIF
import VideoFunctions as VF


def average_frames(Vid, n_sample_frames=100):
    """
    Returns a frame produced by averaging n_sample_frames randomly selected
    frames from video Vid.
    """
    # get number of frames from video
    n_frames = Vid.get(7)
    
    # randomly select frames without replacement
    frame_num_arr = np.random.choice(np.arange(n_frames), size=n_sample_frames,
                                     replace=False)
    
    # sum pixel values from all frames
    first_frame = VF.extract_frame(Vid, frame_num_arr[0])
    pixel_sum = first_frame.astype(int)
    for f in frame_num_arr[1:]:
        pixel_sum += VF.extract_frame(Vid, f)
        
    # divide by number of frames summed to get mean
    pixel_mean = pixel_sum / n_sample_frames
    average_frame = pixel_mean.astype('uint8')
    
    return average_frame


def calculate_stream_width(im, left, right):
    """
    Calculates the width (vertical spanning distance) of outline of image of
    stream from given image, only considering the pixels within the given left
    and right limits.
    """
    # sum all stream widths, average later by dividing by number of summations
    streamWidthSum = 0
    streamWidthSqSum = 0
    colCt = 0
    cutCols = [] # list of columns that were cut off by mask
    # measure angle
    firstXY = []
    lastXY = []
    # loop through column indices to determine average stream width in pixels
    for p in range(left, right):
        # extract current column from masked image
        col = im[:,p,1]
        # skip if column is masked
        if np.sum(col) == 0:
            continue
        # locate saturated pixels
        is255 = col==255
        # if more saturated pixels than just the upper and lower bounds of the contour, stop analysis
        if np.sum(is255) > 2:
            print('Error: more than 2 entries = 255.')
            continue
        elif np.sum(is255) < 2:
            cutCols += [p]
            continue
        # if only upper and lower bounds of contour are saturated, measure separation in pixels
        else:
            # tally the number of columns that are in proper format
            colCt += 1
            # rows with saturated pixels
            rows = np.where(is255)[0]
            if len(firstXY) == 0:
                firstXY = (p, max(rows))
            lastXY = (p, max(rows))
            # calculate width of stream by taking difference of locations of saturated pixels
            streamWidth = np.diff(rows)[0]
            streamWidthSum += streamWidth
            streamWidthSqSum += streamWidth**2
    # print range of columns cut off by mask
    if len(cutCols) > 0:
        print('Error: part of contour cut out by mask, columns from ' + \
        str(min(cutCols)) + ' to ' + str(max(cutCols)))

   # divide sum by number of elements to calculate the mean width
    print('columns counted = ' + str(colCt))
    if colCt == 0:
        streamWidthMean = 0
        streamWidthStDev = 0
    else:
        streamWidthMean = float(streamWidthSum) / colCt
        streamWidthStDev = np.sqrt(float(streamWidthSqSum) / colCt - streamWidthMean**2)

    # correct for angle
    th = np.arctan((lastXY[1]-firstXY[1])/(lastXY[0]-firstXY[0]))
    print('theta = ' + str(th))
    angleCorr = np.cos(th)
    streamWidthMean *= angleCorr
    streamWidthStDev *= angleCorr
    
    return streamWidthMean, streamWidthStDev

def clean_up_bw_im(imBin, selem, minSize):
    """
    cleans up given binary image by closing gaps, filling holes, smoothing
    fringes, and removing specks
    """
    # close region in case of gaps
    closed = skimage.morphology.binary_closing(imBin, selem=selem)
    # fill holes again
    filled = ndimage.morphology.binary_fill_holes(closed)
    # remove fringes
    noFringes = skimage.morphology.binary_opening(filled, selem=selem)
    # remove small objects
    imClean = skimage.morphology.remove_small_objects(noFringes.astype(bool), min_size=minSize)

    return imClean


def create_and_apply_mask(im, shape, message=''):
    """
    Has user create mask and applies it to given image.
    """
    # obtain vertices of user-defined mask from clicks
    maskVertices = UIF.define_outer_edge(im,'rectangle',
                                     message=message)
    # create mask from vertices
    mask, maskPts = create_polygon_mask(im, maskVertices)
    # mask image so only region around inner stream is shown
    imMasked = mask_image(im, mask)

    return imMasked


def create_circular_mask(image,R,center):
    """
    Create 2D mask (boolean array) with same dimensions as input image
    where everything outside of the circle is masked.
    """
    # Calculate the number of points needed based on the size of the radius in
    # pixels (2 points per unit pixel)
    nPoints = int(4*np.pi*R)
    # Generate X and Y values of points on circle
    x,y = Fun.generate_circle(R,center,nPoints)
    mask = get_mask(x,y,np.shape(image))

    return mask
 
    
def create_polygon_mask(image,points):
    """
    Create 2D mask (boolean array) with same dimensions as input image
    where everything outside of the polygon is masked.
    """
    # Calculate the number of points needed perimeter of the polygon in
    # pixels (4 points per unit pixel)
    points = np.array(points,dtype=int)
    perimeter = cv2.arcLength(points,closed=True)
    nPoints = int(2*perimeter)
    # Generate x and y values of polygon
    x = points[:,0]; y = points[:,1]
    x,y = Fun.generate_polygon(x,y,nPoints)
    points = [(int(x[i]),int(y[i])) for i in range(nPoints)]
    points = np.asarray(list(unique(points)))
    mask = get_mask(x,y,image.shape)

    return mask, points


def create_rect_mask_data(im,maskFile):
    """
    create mask for an image and save as pickle file
    """
    maskMsg = "Click opposing corners of rectangle outlining desired region."
    # obtain vertices of mask from clicks; mask vertices are in clockwise order
    # starting from upper left corner
    maskVertices = UIF.define_outer_edge(im,'rectangle',
                                         message=maskMsg)
    xMin = maskVertices[0][0]
    xMax = maskVertices[1][0]
    yMin = maskVertices[0][1]
    yMax = maskVertices[2][1]
    xyMinMax = np.array([xMin, xMax, yMin, yMax])
    # create mask from vertices
    mask, maskPts = create_polygon_mask(im, maskVertices)
    print(mask)
    # store mask data
    maskData = {}
    maskData['mask'] = mask
    maskData['xyMinMax'] = xyMinMax
    # save new mask
    with open(maskFile,'wb') as f:
        pkl.dump(maskData, f)

    return maskData

def create_polygonal_mask_data(im,maskFile):
    """
    create mask for an image and save as pickle file
    """
    # create mask
    points = UIF.define_outer_edge(im,'polygon',message='click vertices')
    mask, points = create_polygon_mask(im, points)
    # store mask data
    maskData = {}
    maskData['mask'] = mask
    maskData['points'] = points
    # save new mask
    with open(maskFile,'wb') as f:
        pkl.dump(maskData, f)

    return maskData


def create_mask_data(image,maskFile):
    """
    Create a dictionary containing the mask information from the given image
    and save to a pickle file.
    """
    # parse input
    plt.gray()
    plt.close()
    message = 'Click on points at the outer edge of the disk for mask'
    R,center = UIF.define_outer_edge(image,'circle',message)
    mask1 = create_circular_mask(image,R,center)
    plt.figure()
    plt.imshow(image)
    image = mask_image(image,mask1)
    plt.figure()
    plt.imshow(image)
    message = 'Click points around the nozzle and tubing for mask'
    points = UIF.define_outer_edge(image,'polygon',message)
    mask2, temp = create_polygon_mask(image,points)
    # invert polygon mask and combine with circle mask
    mask2 = (mask2 != True)
    mask = (mask2 == mask1)*mask1
    image = mask_image(image,mask)
    Fun.plt_show_image(image)

    maskData = {}
    maskData['mask'] = mask
    maskData['diskMask'] = mask1
    maskData['nozzleMask'] = mask2
    maskData['diskCenter'] = center
    maskData['diskRadius'] = R
    maskData['nozzlePoints'] = points
    maskData['maskRadius'] = R

    with open(maskFile,'wb') as f:
        pkl.dump(maskData,f)

    return maskData


def dilate_mask(mask, size=6, iterations=2):
    """
    Deprecated name for "erode_mask"
    """
    return erode_mask(mask, size=size, iterations=iterations)

def erode_mask(mask,size=6,iterations=2):
    """
    Increase the size of the masked area by dilating the mask to block
    additional pixels that surround the existing blacked pixels.
    """
    kernel = np.ones((size,size),np.uint8)
    dilation = cv2.erode(np.uint8(mask),kernel,iterations=iterations)
    mask = dilation.astype(bool)

    return mask


def get_auto_thresh(im, tol=1):
    """
    Uses "automatic thresholding" as described at
    https://en.wikipedia.org/wiki/Thresholding_(image_processing)
    """
    thresh = np.mean(im)
    ret, imThresh = cv2.threshold(im,thresh,255,cv2.THRESH_BINARY)
    whiteMean = np.mean(im[imThresh])
    blackMean = np.mean(im[np.logical_not(imThresh)])
    newThresh = np.mean([whiteMean, blackMean])
    while np.abs(newThresh-thresh) > tol:
        thresh = newThresh
        ret, imThresh = cv2.threshold(im,thresh,255,cv2.THRESH_BINARY)
        whiteMean = np.mean(im[imThresh])
        blackMean = np.mean(im[np.logical_not(imThresh)])
        newThresh = np.mean([whiteMean, blackMean])

    return newThresh

    
def get_auto_thresh_hist(im, frac=0.1):
    """
    returns a suggested value for the threshold to apply to the given image to
    distinguish foreground from background/feature from noise.
    """
    # compute histogram for image
    values, counts = np.unique(im, return_counts=True)
    histMax = np.max(counts)
    # only consider values below fraction of peak above the histogram peak
    belowFrac = counts < frac*histMax
    # if threshold is too low, recurse with a larger fraction
    if np.sum(belowFrac) == 0:
        return get_auto_thresh_hist(im, frac=(frac*1.2))
    crossings = np.logical_xor(belowFrac,np.roll(belowFrac,-1))
    lastCrossing = np.where(crossings)[0][-1]
    # return the first value that dips below the fraction of the peak
    thresh = values[lastCrossing]

    return thresh


def get_auto_thresh_double_gaussian(im, showPlot=False, nSigma=3.0):
    """
    """
    # collect counts of pixel values for histogram
    values, counts = np.unique(im, return_counts=True)
    # apply median filter to counts for smoother histogram, easier to process
    counts = medfilt(counts, kernel_size=5) 
    # find max value with non-negligible amount of counts (above noise)
    maxCt = np.max(counts)
    # find second peak by scanning crossings at lower counts
    reachedValley = False
    for i in range(0,100):
        ct = maxCt*i/100.0
        cr = np.where(Fun.get_crossings(counts, ct))[0]
        # reached valley between two peaks
        if len(cr) >= 4:
            reachedValley = True
        # if only 2 crossings (i.e. at level where there is only one peak)
        if reachedValley and len(cr) <= 2:  
            while len(cr) != 4:
                i-=1
                ct = maxCt*i/100.0
                cr = np.where(Fun.get_crossings(counts, ct))[0]
            break
    ctPeak1 = np.max(counts[cr[0]:cr[1]])
    crPeak1 = np.where(counts==ctPeak1)[0]
    iPeak1 = Fun.choose_middle(crPeak1[np.logical_and(crPeak1 > cr[0], crPeak1 <cr[1])])
    ctPeak2 = np.max(counts[cr[2]:cr[3]])
    crPeak2 = np.where(counts==ctPeak2)[0]
    iPeak2 = Fun.choose_middle(crPeak2[np.logical_and(crPeak2 > cr[2], crPeak2 <cr[3])])
    # guess amplitudes based on max of each peak, and means as their indices
    a1G = ctPeak1
    mu1G = values[iPeak1]
    a2G = ctPeak2
    mu2G = values[iPeak2]    

    # split indices for each peak
    ctValley = np.min(counts[iPeak1:iPeak2])
    crValley = np.where(counts==ctValley)[0]
    iValley = Fun.choose_middle(crValley[np.logical_and(crValley > iPeak1, crValley < iPeak2)])
    counts1 = counts[:iValley]
    values1 = values[:iValley]
    counts2 = counts[iValley:]
    values2 = values[iValley:]
    ### guess standard deviations based on full width at half maximum ###
    # find indices of points where histogram crosses half maximum of tallest peak
    iCrossing1 = np.where(Fun.get_crossings(counts1, (a1G-ctValley)/2.0+ctValley))[0]
    # append max and minimum values to indices of crossings for edge cases
    iCrossing1 = np.concatenate((np.array([0]), iCrossing1,np.array([len(values1)-1])))
    # index of crossing immediately before peak (assumes no wild fluctuations)
    iLeft1 = Fun.choose_middle([i for i in range(len(iCrossing1)-1) if iCrossing1[i] <= iPeak1 \
              and iCrossing1[i+1] >= iPeak1])
    iRight1 = iLeft1 + 1
    # full width at half maximum is difference in values at half max on either
    # side of peak
    fwhm1 = values1[iCrossing1[iRight1]] - values1[iCrossing1[iLeft1]]
    
    # find indices of points where histogram crosses half maximum of tallest peak
    iCrossing2 = np.where(Fun.get_crossings(counts2, (a2G-ctValley)/2.0+ctValley))[0]
    # append max and minimum values to indices of crossings for edge cases
    iCrossing2 = np.concatenate((np.array([0]), iCrossing2,np.array([len(values2)-1])))
    # shift index of peak2 to match indices of counts2
    iPeakShifted2 = iPeak2 - len(counts1)
    # index of crossing just below tallest peak (assumes no wild fluctuations)
    iLeft2 = Fun.choose_middle([i for i in range(len(iCrossing2)-1) if \
                                iCrossing2[i] <= iPeakShifted2 and iCrossing2[i+1] >= iPeakShifted2])
    # full width at half maximum is difference in values at half max on either
    # side of peak
    iRight2 = iLeft2 + 1
    fwhm2 = values2[iCrossing2[iRight2]] - values2[iCrossing2[iLeft2]]
    
    # guess for standard deviation is full width at half maximum of tallest peak
    sG1 = fwhm1/2.0
    sG2 = fwhm2/2.0
    
#    print('a1G ' + str(a1G) + ' mu1G ' + str(mu1G) + ' sG1 ' + str(sG1) +\
#          ' a2G ' + str(a2G) + ' mu2G ' + str(mu2G) + ' sG2 ' + str(sG2))
    # Least squares fit. Starting values found by inspection.
    paramsG =  np.array([a1G, mu1G, sG1, a2G, mu2G, sG2])
    print(paramsG)
    result = least_squares(lambda params:Fun.double_gaussian_fit(values, \
                             counts, params), paramsG, bounds=(np.array([0,0,0,0,0,0]),\
                               np.array([np.inf, 255, 255, np.inf, 255, 255])))
    params = result.x
    # print termination reason
    print(result.message)
#    print('params are ' + str(params))
    # extract parameters and assign them to the taller and smaller peaks accordingly
    (a1, mu1, s1, a2, mu2, s2) = params
    # find lower-mean peak by offsetting by 3 if mu1 > mu2
    offsetLow = 3*(mu1>mu2)
    muLow = params[1+offsetLow]
    sLow = params[2+offsetLow]
    offsetHigh = 3*(mu1<mu2)
    muHigh = params[1+offsetHigh]
    sHigh = params[2+offsetHigh]
    # if lower mean is below zero, cut out both peaks (cut a little more)
    if muLow < 0:
        print('lower mean is below zero.')
        thresh = muHigh + (nSigma+1)*np.abs(sHigh)
    # if gaussians overlap, threshold excludes lower-mean peak
    elif Fun.overlapping_gaussians(params, nSigma=nSigma):
        print('gaussians overlap')
        thresh = muLow + nSigma*np.abs(sLow) # absolute value in case stdev is negative
    # if gaussians do not overlap, take intersection or point that excludes lower-mean peak
    else:
        # find intersection points
        valsIntersection = np.array(Fun.gaussian_intersections(params)) 
        # find intersections in between means
        iBetween = np.logical_and(valsIntersection < muHigh, valsIntersection > muLow)
        valsBetween = valsIntersection[iBetween]
        print(valsBetween)
        # if no intersection, then threshold is two stdevs above lower mean
        if len(valsBetween) == 0:
            thresh = muLow + nSigma*sLow
            print('no intersection between means')
        else:
            # threshold is larger of intersection between peaks of two gaussians
            thresh = np.max(valsBetween)
            print('threshold is at intersection')
    # plot data, fit and threshold if desired
    if showPlot:
        plt.figure()
        plt.plot(values, counts, label='data')
        yGauss = Fun.double_gaussian(values, params)
        plt.plot(values, yGauss, label='double gauss fit')
        yLim = plt.ylim()
        plt.plot(np.array([thresh, thresh]), yLim, 'r--', label='thresh')
        plt.legend(loc='best')
        plt.xlabel('values')
        plt.ylabel('counts')
        plt.title('comparing counts of pixel values to double gaussian fit')
    
    return int(thresh)



def get_auto_thresh_plateau(im, frac=0.1, recMult=1.1):
    """
    returns a suggested value for the threshold to apply to the given image to
    distinguish foreground from background/feature from noise based on the
    plateau of the histogram of pixel counts.

    frac = fraction of peak number of counts (excluding for saturated pixels)
    recMult = value to increase frac by if too low per recursion
    """
    # compute histogram for mean values of rows of image
    values, counts = np.unique(im, return_counts=True)
    # find max of lower peak
    histMax = np.max(counts)
    iMax = np.argmax(counts)
    # only consider values above max of lower peak
    values = values[iMax:]
    counts = counts[iMax:]
    # locate crossings across threshold for number of counts
    crossings = Fun.get_crossings(counts, frac*histMax)
    if np.sum(crossings) == 0:
        return get_auto_thresh_plateau(im, frac=(frac*recMult), recMult=recMult)
    iSeq, lengthSeq = Fun.longest_sequence(crossings)
    # return the first value that dips below the fraction of the peak
    thresh = values[iSeq]

    return thresh

def get_auto_thresh_rows(im, frac=0.1):
    """
    returns a suggested value for the threshold to apply to the given image to
    distinguish foreground from background/feature from noise.
    """
    # compute histogram for mean values of rows of image
    values, counts = np.unique(np.mean(im,1).astype('uint8'), return_counts=True)
    histMax = np.max(counts)
    # only consider values below fraction of peak above the histogram peak
    belowFrac = counts < frac*histMax
    if np.sum(belowFrac) == 0:
        return get_auto_thresh_rows(im, frac=(frac*1.2))
    # identify when the histogram crosses the given fraction of the histogram peak
    crossings = np.logical_xor(belowFrac,np.roll(belowFrac,-1))
    lastCrossing = np.where(crossings)[0][-1]
    # return the first value that dips below the fraction of the peak
    thresh = values[lastCrossing]

    return thresh

def get_edgeX(outlinedIm, channel='g', imageType='rgb'):
    """
    Returns the left and right indices marking the edges of the
    outlined region marked in the given channel (default green).
    """
    c = get_channel_index(channel, imageType=imageType)
    isEdgeX = np.where(outlinedIm[:,:,c]==255)[1]
    left = np.min(isEdgeX)
    right = np.max(isEdgeX)
    return left, right


def get_mask(X,Y,imageShape):
    """
    Converts arrays of x- and y-values into a mask. The x and y values must be
    made up of adjacent pixel locations to get a filled mask.
    """
    # Take only the first two dimensions of the image shape
    if len(imageShape) == 3:
        imageShape = imageShape[0:2]
    # Convert to unsigned integer type to save memory and avoid fractional
    # pixel assignment
    X = X.astype('uint16')
    Y = Y.astype('uint16')

    #Initialize mask as matrix of zeros
    mask = np.zeros(imageShape,dtype='uint8')
    # Set boundary provided by x,y values to 255 (white)
    mask[Y,X] = 255
    # Fill in the boundary (output is a boolean array)
    mask = ndimage.morphology.binary_fill_holes(mask)

    return mask

def get_roi(im, roiLims, coordinateFormat='xy'):
    """
    returns section of image delimited by limits of region of interest
    roiLims gives
    Coordinate format:
    'xy': [xMin, xMax, yMin, yMax]
    'rc': [rMin, rMax, cMin, cMax]
    """
    if coordinateFormat=='xy':
        c1, c2, r1, r2 = roiLims
    elif coordinateFormat=='rc':
        r1, r2, c1, c2 = roiLims
    else:
        print('unrecognized coordinateFormat in IPF.get_roi.')
        return []
    if len(im.shape)==3:
        return im[r1:r2,c1:c2,:]
    elif len(im.shape)==2:
        return im[r1:r2,c1:c2]
    else:
        print('image is in improper format; not 2 or 3 dimensional (IPF.get_roi).')
        return []


def mask_image(image, mask):
    """
    Returns image with all pixels outside mask blacked out
    mask is boolean array or array of 0s and 1s of same shape as image
    """
    # Apply mask depending on dimensions of image
    temp = np.shape(image)
    maskedImage = np.zeros_like(image)
    if len(temp) == 3:
        for i in range(3):
            maskedImage[:,:,i] = mask*image[:,:,i]
    else:
        maskedImage = image*mask

    return maskedImage


def project_im(im, color):
    """
    Projects the given 3D image array onto the given color.
    """
    return (np.tensordot(im, color, axes=([2,0]))).astype('uint8')

def reduce_mask_radius(maskData,fraction):
    """
    Reduce the radius of the mask that covers the outer edge of the disk to the
    requested percentage.
    """
    # parse input
    R = maskData['diskRadius']
    center = maskData['diskCenter']
    mask = maskData['mask']
    # Redine the outer circular edge
    mask1 = create_circular_mask(mask,R*fraction,center)
    mask = mask*mask1
    # Save new mask data
    maskData['mask'] = mask
    maskData['diskMask'] = mask1
    maskData['maskRadius'] = R*fraction

    return maskData


def rgb_gauss(im, selem):
    """
    applies gaussian filter to an rgb image (i.e., applies it
    to each channel)
    """
    imCopy = np.copy(im)
    for i in range(3):
        imCopy[:,:,i] = filters.rank.mean(im[:,:,i], selem)
    return imCopy


def scale_by_brightfield(im, bf):
    """
    scale pixels by value in brightfield
    """
    # convert to intensity map scaled to 1.0
    bf1 = bf.astype(float) / 255.0
    # scale by bright field
    imCorrected = np.divide(im,bf1)
    # rescale result to have a max pixel value of 255
    imCorrected *= 255.0/np.max(imCorrected)
    # change type to uint8
    imCorrected = imCorrected.astype('uint8')

    return imCorrected

def define_homography_matrix(image,hFile,scale=1.1):
    """
    Use an image of the wafer chuck from a given camera orientation to create
    a homography matrix for transforming the image so that the wafer appears
    to be a perfect circle with the center at the center of the image.

    The user will need to identify the wafer chuck posts in the image by
    clicking on the center of the small cylinder on the top of the post
    in the following order:
    First, the post farthest to the right, then proceeding in clockwise fashion
    around the edge of the chuck. The first 6 points will be used.
    If the first post is different it will rotate the image and if the
    direction is changed it will mirror the image.
    """
    # Get the post locations in the original image
    message = 'Click on the center of the 6 posts holding the disk. \n' + \
        'Start at the right side and proceed clockwise. \n' + \
        'Center click when finished'
    oldPoints = UIF.define_outer_edge(image,'polygon',message)
    oldPoints = np.array(oldPoints[:6])
    # Get the size of the original image and use it define the size of the
    # disk in the new image
    temp = np.shape(image)
    R = scale*Fun.get_distance(oldPoints[0,:],oldPoints[3,:])/2.0
    center = [np.mean(oldPoints[:,0]),np.mean(oldPoints[:,1])]
    t0 = Fun.get_angle([center[0]+10,center[1]],center,oldPoints[0])
    print(str(t0))
    # Define the locatoins of the posts in the new image
    x,y = Fun.generate_circle(R,center,7,t0)
    imageCenter = [temp[1]/2.0,temp[0]/2.0]
    dX = center[0] - imageCenter[0]
    dY = center[1] - imageCenter[1]
    x -= dX
    y -= dY
    newPoints = np.array([(x[i],y[i]) for i in range(6)])
    # Define the homography matrix
    H,M = cv2.findHomography(oldPoints,newPoints)
    # View the resulting transformation
    newImage = cv2.warpPerspective(image,H,(temp[1],temp[0]))
    plt.figure()
    plt.subplot(121)
    Fun.plt_show_image(image)
    plt.plot(x,y,'go',oldPoints[:,0],oldPoints[:,1],'ro')
    plt.subplot(122)
    Fun.plt_show_image(newImage)
    plt.plot(x,y,'go',oldPoints[:,0],oldPoints[:,1],'ro')
    plt.savefig('../Figures/Homography.png')
    plt.ginput()

    with open(hFile,'wb') as f:
        pkl.dump(H,f)

    return H


def get_channel(im, channel, imageType='rgb', rgb=np.array([0,0,0]),
                subtRGB=np.array([0,0,0])):
    """
    Returns one channel of a color image (i.e., red, green, or blue of an rgb
    image). If image is assumed rgb by default. Channel should indicate the
    desired color by one lowercase letter (e.g., 'b' for the blue channel).
    """
    assert(is_color(im))
    # project image onto a custom color
    if channel == 'custom':
        scale = np.sqrt(np.sum(rgb**2))
        # normalized projection vector
        proj = rgb / scale
        # if not subtracting another color, proceed with projection
        if np.sum(subtRGB) == 0:
            channel = np.dot(im, proj).astype('uint8')
        # otherwise subtract the given color
        else:
            scaleSubt = np.sqrt(np.sum(subtRGB**2))
            subt = subtRGB / scaleSubt
            # scale cross product to be normalized to length 1
            crossProd = np.cross(proj, subt)
            crossProd /= np.sqrt(np.sum(crossProd**2))
            # change of basis matrix to projected color, subtracted, and their cross product
            B = np.stack([proj, subt, crossProd],1)
            BInv = np.linalg.inv(B)
            # change basis of image's rgb values to new colors
            imProj = np.matmul(im, BInv)
            # get "red" (first) channel from projected image
            channel = get_channel(imProj, 'r')
            # scale image so max value is saturated (255)
            maxVal = np.max(channel)
            channel *= 255.0/maxVal
            # change type to unsigned 8-bit int for images (0-255)
            channel = channel.astype('uint8')
    else:
        c = get_channel_index(channel, imageType)
        channel = im[:,:,c]

    return channel


def get_channel_index(channel, imageType='rgb'):
    """
    Returns the index of a given color channel. Color channel must be given by
    the first letter in the color name, lowercase.
    """
    if imageType == 'rgb':
        channelDict = {'r':0, 'g':1, 'b':2}
    # bgr is used by cv2
    elif imageType == 'bgr':
        channelDict = {'b':0, 'g':1, 'r':2}
    else:
        print('imageType ' + imageType + ' not recognized.')

    return channelDict[channel]


def get_contour_bw_im(imBin, showIm, lineWidth=1):
    """
    returns image showing only a 1-pixel thick contour enclosing largest object
    in the binary image given.
    """
    #find edge using contour finding
    _, contours, hierarchy = cv2.findContours(imBin.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # return blank image if no contours
    if len(contours) == 0:
        return np.zeros_like(imBin,dtype='uint8')
    cntMax = sorted(contours, key=cv2.contourArea, reverse=True)[0] # get largest contour
    lenChar = np.sqrt(imBin.size) # length scale = sqrt num pixels
    nPtsCnt = int(4*lenChar)
    # generate continuous array of points of largest contour
    x,y = Fun.generate_polygon(cntMax[:,0,0], cntMax[:,0,1], nPtsCnt)
    xyPairs = list(zip(x,y))
    cntMaxZip = np.array(xyPairs,dtype=int)
    # create image of contour
    imCnt = np.zeros_like(imBin,dtype='uint8')
    cv2.drawContours(imCnt, [cntMaxZip], -1, 255, 1)
    # dilation
    assert(lineWidth >= 1, 'Line width must be greater than or equal to 1.')
    for i in range(lineWidth-1):
        imCnt = skimage.morphology.binary_dilation(imCnt)
    
    return imCnt


def get_negative(im, maxPixelValue=255):
    """
    Returns negative of given image. Default is 255-scale image.
    """
    return maxPixelValue - im


def is_color(im):
    """
    Returns true if image is a color (3-dimensional) image and false if not.
    """
    # color images are three-dimensional matrices, so shape must have 3 dimensions
    return len(im.shape)==3


def mask_xy(xyvals,mask):
    """
    Returns only the xyvals of points that are not blocked by the mask
    """
    maskedEdgeLocations = np.array([[z[1],z[0]] for z in xyvals if mask[z[0],z[1]]])

    return maskedEdgeLocations

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

def scale_image(im,scale):
    """
    Scale the image by multiplicative scale factor "scale".
    """
    temp = im.shape
    im = cv2.resize(im,(int(scale*temp[1]),int(scale*temp[0])))

    return im

def show_im(im, title='', showCounts=False, tFS=24, pixPerMicron=0):
    """
    Shows image in new figure with given title
    """
    # open a new figure
    plt.figure()
    if showCounts:
        # plot image
        plt.subplot(121)
        plt.imshow(im)
        plt.title(title)
        # count pixel values present in image
        values, counts = np.unique(im, return_counts=True)
        # plot counts
        plt.subplot(122)
        plt.plot(values, counts)
        plt.title('pixel value counts', fontsize=tFS)
        # display image
        plt.show()
    else:
        xMax = im.shape[1]
        yMax = im.shape[0]
        # if given a conversion for pixels to microns
        if pixPerMicron > 0:
            xMax /= pixPerMicron
            yMax /= pixPerMicron
        # plot image
        plt.imshow(im, extent=[0, xMax, 0, yMax])
        plt.title(title, fontsize=tFS)
        # add axis label if shown in microns
        if pixPerMicron > 0:
            plt.xlabel('um')
            plt.ylabel('um')            
        plt.show()


def subtract_images(frame,refFrame):
    """
    Subtracts "prevFrame" from "frame", returning the absolute difference.
    This function is written for 8 bit images.
    """
    if frame.shape != refFrame.shape:
        raise('Error: frames must be the same shape for subtraction')

    # Convert frames from uints to ints for subtraction
    frame = frame.astype(int) # Allow for negative values
    refFrame = refFrame.astype(int) # Allow for negative values

    # Take absolute difference and convert to 8 bit
    result = abs(frame - refFrame)
    result = result.astype('uint8')

    return result

def superimpose_bw_on_color(imColor, imBW, roiLims, channel, imageType='rgb',
                            coordinateFormat='xy'):
    """
    Superimposes a black and white image onto one of the channels of a color
    image within a region of interest. The color of the channel can be selected.
    Only returns whether the operation was successful (there was something in imBW
    and it was superimposed) or not.
    """
    # copy image so it can be edited
    imColor = np.copy(imColor)
    # check if there is anything in the black and white image
    if np.sum(imBW) == 0:
        print('nothing in black and white image')
        return np.array([]), False
    # extract limits of region of interest
    if coordinateFormat == 'xy':
        c1, c2, r1, r2 = roiLims
    elif coordinateFormat == 'rc':
        r1, r2, c1, c2 = roiLims
    else:
        print('unrecognized coordinate format; not xy (default) or rc.')
    # hold bw image in new frame same size as color image but 1 channel deep
    imFullBW = np.zeros_like(imColor, 'uint8')[:,:,0]
    imFullBW[r1:r2,c1:c2] = imBW
    imSuperimposed = imColor
    # set all pixels from bw image to saturation (255)
    imSuperimposed[imFullBW.astype(bool), get_channel_index(channel, imageType)] = 255

    # superimposing was successful
    return imSuperimposed, True

def threshold_im(im, thresh=-1, c=None):
    """
    Applies a threshold to the image and returns black-and-white result.
    """
    nDims = len(im.shape)
    if nDims == 3:
        if not c:
            brightestChannel = -1
            brightestPixelVal = -1
            for i in range(nDims):
                brightestCurr = np.max(im[:,:,i])
                if brightestCurr > brightestPixelVal:
                    brightestChannel = i
                    brightestPixelVal = brightestCurr
        else:
            brightestChannel = c
        im = np.copy(im[:,:,brightestChannel])
    if thresh == -1:
        # Otsu's thresholding
        ret, threshIm = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        ret, threshIm = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)

    return threshIm
 
def filter_frame(frame):
    """
    Apply the prescribed filter to the video frame to remove noise.
    """

    denoised = ndimage.gaussian_filter(frame,0.03)

    return denoised

def fft_image(image):
    """
    Take the fourier transform of an image and return the frequency spectrum
    with the zero mode at the center.
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    return fshift

def invfft_image(fshift):
    """
    Convert fourier transformed image back to real space.
    """
    f = np.fft.ifftshift(fshift)
    image = np.uint8(np.abs(np.fft.ifft2(f)))

    return image

def apply_butterworth_filter(image, cutoff, n):
    """
    Take the given image and apply a butterworth filter using Fourier
    transforms.
    """
    # Transform the image to fourier space
    size = np.shape(image)
    f = fft_image(image)
    # Parse filter parameters
    if (cutoff <= 0) or (cutoff > 0.5):
        print('cutoff frequency must be between (0 and 0.5]')
        cutoff = 0.5
    n = np.uint8(n)
    # Create the filter
    x,y = np.meshgrid(np.linspace(0,1,size[1]),np.linspace(0,1,size[0]))
    x = x - 0.5
    y = y - 0.5
    radius = np.sqrt(x**2.0 + y**2.0)
    butter = 1.0/(1.0 + (radius/cutoff)**(2.0*n))
    #Apply the filter and transform back
    f = f*butter
    filtered = invfft_image(f)

    return filtered

def apply_local_otsu(image,radius=50):
    """

    """
    selem = disk(radius)
    newImage = np.uint16(image)
    local_otsu = filters.rank.otsu(newImage,selem)
    newImage = np.uint8(newImage >= local_otsu)

    return newImage

def bgr2rgb(frame):
    """
    Converts BGR frame (received by high-speed camera) to RGB
    """
    temp = np.ndarray.copy(frame)
    frame = np.dstack((temp[:,:,2],temp[:,:,1],temp[:,:,0]))
    
    return frame

def mean_filter(frame, selem):
    """
    Performs mean filter on each of the color channels of an 3-channel image.
    """
    shape = frame.shape
    if len(shape) == 2:
        frame = skimage.filters.rank.mean(frame, selem)
    elif len(shape) == 3:
        for i in range(len(frame[0,0,:])):
            frame[:,:,i] = skimage.filters.rank.mean(frame[:,:,i], selem)
    else:
        print("Frame is not either a 2D or 3D array and was not filtered.")
        
    return frame

def process_frame(frame,ref,wettedArea,theta,threshold1,threshold2):
    """
    Compare the frame of interest to a reference frame and already known
    wetted area and determine what additional areas have becomed wetted using
    the absolute difference and the difference in edge detection.
    """
    # For testing and optimizing, use these flags to turn off either step
    simpleSubtraction = True
    edgeSubtraction = True
    cutoff = 254

    # Initialize storage variables
    image1 = np.zeros_like(frame)
    image2 = np.zeros_like(frame)
    tempRef = np.zeros_like(frame)
    comp1 = np.zeros_like(ref)
    comp2 = np.zeros_like(ref)

    # Generate comparison data between reference and image
    if simpleSubtraction:
        image1 = subtract_images(frame,ref)
    if edgeSubtraction:
        tempFrame = np.uint8(filters.prewitt(frame)*255)
        tempRef = np.uint8(filters.prewitt(ref)*255)
        image2 = subtract_images(tempFrame,tempRef)

    # Prepare matrices for thresholding the results
    # Apply different thresholds at different intensities
    comp1[:] = threshold1
    comp2[:] = threshold2
#    comp1[ref>=30] = threshold1
#    comp1[ref<30] = 2*threshold1
#    comp2[tempRef<=128] = threshold2
#    comp2[tempRef>128] = 2*threshold2
#    comp2[tempRef<30] = threshold2*.75

    # Convert the results to 8 bit images for combining
    image1 = np.uint8((image1 > comp1)*255)
    image2 = np.uint8((image2 > threshold2)*255)
#    wettedArea = np.uint8(wettedArea*255)

    # Depending on whether or not the disk is rotating, apply thresholding
    if theta != 0:
        image1 = rotate_image(image1,theta,size=image1.shape)
        image2 = rotate_image(image2,theta,size=image2.shape)

    wettedArea = wettedArea + (image1>cutoff) + (image2>cutoff)

    # Fill holes in the wetted area
#    wettedArea = ndimage.morphology.binary_fill_holes(wettedArea)

    return wettedArea

def get_perimeter(wettedArea):
    """
    Find the perimeter of the wetted area which is identified as the largest
    region of contiguous wetted area.
    """
    contours = cv2.findContours(np.uint8(wettedArea*255),cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)[0]

#    data1 = max(contours,key=len)
    data2 = max(contours,key=cv2.contourArea)
    data = data2.squeeze()
    data = np.reshape(data,(-1,2))

    return data

def float2uint8(im, subtMin=True):
    """
    Converts float image scaled from 0 to 1 (like output of skimage's gaussian
    filter) to a 255-scale uint8 image.
    """
    # scale brightness by subtracting minimum to go from 0 to 1
    if subtMin:
        im = (im - im.min()) / (im.max() - im.min())
    return (255*im).astype('uint8')

def uint82float(im, subtMin=True):
    """
    Converts uint8 image scaled from 0 to 255 to an image of floats scaled from
    0 to 1, which is used in scipy.ndimage.filters. subtMin subtracts the minimum
    value to scale the image as much as possible.
    """
    if subtMin:
        result = (im.astype(float) - im.min()) / (im.max() - im.min())
    else:
        result = im.astype(float)/255.0
    
    return result

def highlight_bubbles(frame, bkgd, thr, selem=skimage.morphology.disk(5), min_size=20):
    """
    Performs background subtraction, thresholding, and cleanup to return black-
    and-white image of bubbles in frame.
    """
    # subtract images
    deltaIm = cv2.absdiff(frame, bkgd)
    # threshold
    thrIm = threshold_im(deltaIm, thr)
    # smooth out thresholded image
    closedIm = skimage.morphology.binary_closing(thrIm, selem=selem)
    # remove small objects
    bubbles = skimage.morphology.remove_small_objects(closedIm.astype(bool), min_size=min_size)
    # convert to uint8
    bubbles = 255*bubbles.astype('uint8')
    
    return bubbles


def scale_brightness(image):
    """
    Rescales the pixel values of the image such that the highest value is 255
    and the lowest is 0 for maximum contrast.
    """
    maxIntensity = np.max(image)
    minIntensity = np.min(image)
    image -= minIntensity
    image = image.astype(float)
    image *= 255. / (maxIntensity - minIntensity)
    
    return image.astype('uint8')

def union_thresholded_ims(im1, im2, pct1, pct2, showIm=False,
                          title1='image 1', title2='image 2'):
    """
    Returns a binary image that is the union of the thresholded blue channel
    and the thresholded negative of the red channel of the given image. Used
    for locating blue stream in images of sheath flow.
    Threshold is done by taking upper percentile of each channel, with cutoffs
    provided by user (pct1 and pct2).
    """
    # show first image if desired
    if showIm:
        show_im(im1, title1)
    # threshold
    ret, im1Thresh = cv2.threshold(im1,np.percentile(im1, pct1),255,cv2.THRESH_BINARY)
    if showIm:
        show_im(im1Thresh, title1 + ' Threshold')
    if showIm:
        show_im(im2, title2)
    # threshold
    ret, im2Thresh = cv2.threshold(im2,np.percentile(im2, pct2),255,cv2.THRESH_BINARY)
    if showIm:
        show_im(im2Thresh, title2 + ' Threshold')
    # combine thresholded images (effectively an 'and' operation)
    imUnion = np.multiply(im1Thresh, im2Thresh)
    if showIm:
        show_im(imUnion, 'Union of thresholded ' + title1 + ' and ' + title2)

    return imUnion


def union_thresholded_ims_cutoff(im1, im2, thresh1, thresh2, showIm=False,
                          title1='image 1', title2='image 2'):
    """
    Returns a binary image that is the union of the thresholded blue channel
    and the thresholded negative of the red channel of the given image. Used
    for locating blue stream in images of sheath flow.
    Threshold is done by taking upper percentile of each channel, with cutoffs
    provided by user (pct1 and pct2).
    """
    # show first image if desired
    if showIm:
        show_im(im1, title1)
    # threshold
    ret, im1Thresh = cv2.threshold(im1,thresh1,255,cv2.THRESH_BINARY)
    if showIm:
        show_im(im1Thresh, title1 + ' Threshold')
    if showIm:
        show_im(im2, title2)
    # threshold
    ret, im2Thresh = cv2.threshold(im2,thresh2,255,cv2.THRESH_BINARY)
    if showIm:
        show_im(im2Thresh, title2 + ' Threshold')
    # combine thresholded images (effectively an 'and' operation)
    imUnion = np.multiply(im1Thresh, im2Thresh)
    if showIm:
        show_im(imUnion, 'Union of thresholded ' + title1 + ' and ' + title2)

    return imUnion

if __name__ == '__main__':
    pass
#    plt.close('all')
#    filePath = '../Data/Prewetting Study/Water_1000RPM_2000mLmin_2000FPS.avi'
#    homographyFile = 'offCenterTopView_15AUG2015.pkl' # Use None for no transform
#    with open(homographyFile) as f:
#        hMatrix = pkl.load(f)
#    Vid = VF.get_video_object(filePath)
#    image = VF.extract_frame(Vid,20,hMatrix=hMatrix)
#    image1 = image[:,:,0]
##    f = fft_image(image)
#
##    image2 = image1
#    image2 = apply_local_otsu(image1,50)
#
#    thresh = ski.filters.threshold_otsu(image1)
#    image3 = image1 >= thresh
#    plt.figure()
#    plt.subplot(1,3,1)
#    plt.imshow(image1,cmap='gray')
#    plt.subplot(1,3,2)
#    plt.imshow(image2,cmap='gray')
#    plt.subplot(1,3,3)
#    plt.imshow(image3,cmap='gray')
