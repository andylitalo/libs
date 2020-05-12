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
        """
        # stores metadata
        self.metadata = {'ID':ID, 'fps':fps, 'frame_dim':frame_dim,
                         'flow_dir':flow_dir, 'pix_per_um':pix_per_um}
        # initializes storage of raw properties
        self.props_raw = {'frame':[], 'centroid':[], 'area':[],'major axis':[],
                          'minor axis':[], 'orientation':[]}
        # initializes storage of processed properties
        self.props_proc = {'average area':None, 'average speed':None, 
                           'average orientation':None, 'average aspect ratio':None}
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

