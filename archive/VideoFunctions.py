# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:52:03 2015

@author: John
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage
import os
import pickle as pkl
import sys

# Custom modules
import ImageProcessingFunctions as IPF
import UserInputFunctions as UIF
import Functions as Fun


def get_RPM_from_file_name(fileName):
    """
    Parse the file name to get the RPM from the experiment.
    """
    name = os.path.split(fileName)[1]
    ind = name.index('RPM')
    rpm = float(name[ind-4:ind])
    return rpm

def get_FPS_from_file_name(fileName):
    """
    Parse the file name to get the FPS from the experiment.
    """
    name = os.path.split(fileName)[1]
    ind = name.index('FPS')
    fps = float(name[ind-4:ind])
    return fps

def get_flowRate_from_file_name(fileName):
    """
    Parse the file name to get the FPS from the experiment.
    """
    name = os.path.split(fileName)[1]
    ind = name.index('mLmin')
    flowRate = float(name[ind-4:ind])
    return flowRate

def extract_frame(Vid,nFrame,hMatrix=None,maskData=None,filterFrame=False,
                  removeBanner=True,center=True,scale=1,angle=0):
    """
    Extracts nFrame'th frame and scales by 'scale' factor from video 'Vid'.
    """
    Vid.set(cv2.CAP_PROP_POS_FRAMES,nFrame)
    ret, frame = Vid.read()
    if not ret:
        print('Frame not read')

    # Scale the size if requested
    if scale != 1:
        frame = IPF.scale_image(frame,scale)

    
    # Apply image transformation using homography matrix if passed
    if hMatrix is not None:
        temp = frame.shape
        frame = cv2.warpPerspective(frame,hMatrix,(temp[1],temp[0]))

    # Apply mask if needed
    if maskData is not None:
        frame = IPF.mask_image(frame,maskData['mask'])
        if center:
            frame = IPF.rotate_image(frame,angle,center=maskData['diskCenter'],
                                     size=frame.shape)

    return frame

def get_t0_frame(vid,hMatrix,maskData,fraction,threshold=20):
    """
    Advance frame by frame from the start of the video and monitor the pixels
    around the center of the disk for the presence of water and return the
    index of the first frame.
    """
    viewResult = False
    viewProgress = False
    tShift = 2 #Number of frames back for comparison

    N = int(vid.get(7)) # number of frames in video
    ref = extract_frame(vid,0,hMatrix,maskData)
    center = maskData['diskCenter']
    R = fraction*maskData['diskRadius']
    x1 = center[0]-R; x2 = center[0]+R
    y1 = center[1]-R; y2 = center[1]+R
    mask = IPF.create_circular_mask(ref,R,center)
    maskData = None

    maxDif = 0

    for i in range(tShift,N):
        ref = extract_frame(vid,i-tShift,hMatrix,maskData)*mask
        frame = extract_frame(vid,i,hMatrix,maskData)*mask
        dif = IPF.subtract_images(ref,frame)


        if viewProgress:
            plt.clf()
            plt.subplot(1,2,1)
            plt.imshow(ref)

            plt.axis([x1,x2,y1,y2])
            plt.gray()
            plt.title('Frame #%i'%(i-tShift))
            plt.subplot(1,2,2)
            plt.imshow(frame)
            plt.axis([x1,x2,y1,y2])
            plt.gray()
            plt.title('Frame #%i'%i)
            plt.tight_layout(pad=0)
            plt.pause(0.001)

        maxDif = np.max(dif)
        if maxDif > threshold:
                t0FrameNumber = i
                break

    if viewResult:
        plt.clf()
        plt.subplot(1,3,1)
        plt.imshow(ref)
        plt.axis([x1,x2,y1,y2])
        plt.gray()
        plt.subplot(1,3,2)
        plt.imshow(frame)
        plt.axis([x1,x2,y1,y2])
        plt.gray()
        plt.subplot(1,3,3)
        frame = extract_frame(vid,i+1,hMatrix,maskData)*mask
        plt.imshow(frame)
        plt.axis([x1,x2,y1,y2])
        plt.gray()
        plt.tight_layout(pad=0)
        plt.pause(0.001)
        plt.pause(3)

    return t0FrameNumber

def get_video_object(filePath):
    """
    Given a file name for a video, returns the video object.
    """
    # Open the video file and get properties
    Vid = cv2.VideoCapture(filePath)
    return Vid

def parse_video_obj(Video):
    """ Use the openCV video object to get the properties of the video file
    that are of interest to the analysis. See the openCV documentation for
    the list of properties.
    INPUT:
        Video = openCV video object
    OUTPUT:
        Props = dictionary of properties of the video
    """
    class Props:
        pass
    Props.Width = Video.get(3)
    Props.Height = Video.get(4)
    Props.fps = Video.get(5)
    Props.NumFrames = Video.get(7)

    return Props

def get_data_structure(vid,fps,RPM,flowRate,dataFile,dataList,hMatrix,maskData,
                       intensityRegion,splashFraction):
    """
    Create or load the data structure for frame-by-frame analysis of spin
    coater data.
    """
    fraction = 0.01
    # If the data file already exists, load it
    if dataFile in dataList:
        with open(dataFile,'rb') as f:
            container = pkl.load(f)
            print('loaded')
        return container

    # Initialize dictionaries
    container = {}
    container['data'] = {}
    container['theta'] = {}
    container['fps'] = fps
    container['RPM'] = RPM
    container['flowRate'] = flowRate
    container['hMatrix'] = hMatrix
    container['maskData'] = maskData
    container['insensityRegion'] = intensityRegion
    container['t0Frame'] = get_t0_frame(vid,hMatrix,maskData,fraction)
    container['splashRemoved'] = [False]*len(splashFraction)
    container['splashIndex'] = 0

    # Initialize the wetted area to completely dry
    ref = extract_frame(vid,0,hMatrix,maskData)
    container['wettedArea'] = (np.zeros(ref.shape)==1)
    container['data'][container['t0Frame']] = (np.zeros(ref.shape)==1)

    with open(dataFile,'wb') as f:
        pkl.dump(container,f)

    return container

def make_png_of_frame():
    plt.close('all')
    frame = 0 #[49,236,247,277,18,217]
    folder = '../Data/Validation/'
    name = 'Aperture2_6000sShutter-0100RPM_1500mLmin_2000FPS.avi'
    maskFile = 'maskData_26OCT2015.pkl'
    homographyFile = 'offCenterTopView_15AUG2015.pkl' # Use None for no transform
    with open(homographyFile) as f:
                hMatrix = pkl.load(f)
    with open(maskFile) as f:
                maskData = pkl.load(f)
    vid = get_video_object(folder+name)
    im = extract_frame(vid,frame,hMatrix,maskData)
    plt.imshow(im)
    plt.gray()
    plt.imsave(folder+name[:-4]+'_frame%04i.png'%(frame),im)

if __name__ == '__main__':
    pass
