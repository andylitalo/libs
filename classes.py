# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:24:27 2020

@author: Andy
the Bubble class stores properties of a bubble from a video, both measured and 
processed post factum.
"""

class Bubble:
    def __init__(self, ID, fps, frame_dim, flow_dir, pix_per_um, props_raw=[]):
        """
        frames: list of ints
        centroids: list of tuples
        areas: same as centroids
        major axes: same as centroids
        minor axes: same as centroids
        orientation: same as centroids
        ID: int
        average flow direction: tuple of two floats normalized to 1
        average area: float
        average speed: float
        fps: float
        
        true centroid is estimated location of centroid adjusted after taking
        account of the bubble being off screen/"on border."
        """
        # stores metadata
        self.metadata = {'ID':ID, 'fps':fps, 'frame_dim':frame_dim,
                         'flow_dir':flow_dir, 'pix_per_um':pix_per_um}
        # initializes storage of raw properties
        self.props_raw = {'frame':[], 'centroid':[], 'area':[],'major axis':[],
                          'minor axis':[], 'orientation':[], 'on border':[]}
        # initializes storage of processed properties
        self.props_proc = {'average area':None, 'average speed':None, 
                           'average orientation':None, 
                           'average aspect ratio':None, 'true centroids':[]}
        # loads raw properties if provided
        if len(props_raw) > 0:
            self.add_props(props_raw)
    
    ###### ACCESSORS ######    
    def get_metadata(self, prop):
        return self.metadata[prop]
    
    def get_prop(self, prop, f):
        """Returns property at given frame f according to dictionary of frames."""
        return self.get_props(prop)[self.get_props('frame').index(f)]
    
    def get_props(self, prop):
        if prop in self.props_raw.keys():
            return self.props_raw[prop]
        elif prop in self.props_proc.keys():
            return self.props_proc[prop]
        else:
            print('Property not found in Bubble object. Returning None.')
            return None
    
    
    ###### MUTATORS ########
    def add_props(self, props):
        """props is properly organized dictionary. Only for raw properties."""
        for key in props.keys():
            if key in self.props_raw.keys():
                self.props_raw[key] += [props[key]]
            else:
                print('Trying to add property not in Bubble.props_raw.')
        keys_not_provided = (list(set(self.props_raw.keys()) - \
                                  set(props.keys())))
        for key in keys_not_provided:
            self.props_raw[key] += [None]
            
            
            
########################## FILEVIDEOSTREAM CLASS ##############################

from threading import Thread
from queue import Queue
import cv2


class FileVideoStream:
    """
    Class for handling threaded loading of videos.
    Source:
        https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
    """
    def __init__(self, path, queueSize=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
        
    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        
        return self
   
    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
   				# if the `grabbed` boolean is `False`, then we have
   				# reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
   				# add the frame to the queue
                self.Q.put(frame)
   
    def read(self):
        # return next frame in the queue
        return self.Q.get()
   
    def more(self):
   		# return True if there are still frames in the queue
        return self.Q.qsize() > 0
   
    def stop(self):
   		# indicate that the thread should be stopped
        self.stopped = True
