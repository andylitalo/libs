# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:52:03 2015

@author: John
"""
import matplotlib.pyplot as plt
import numpy as np
from math import atan2
from scipy import optimize
import cv2
from scipy.interpolate import interp1d

# Custom modules

def actual_flow_rate(flowrate, conversion):
    """
    Converts the purported inner flowrate on the syringe pump to the actual 
    inner flowrate using the conversion factor innerConversion = actual/purported
    """
    return flowrate*conversion
    
    
def add_col(mat, col, colNum):
    matT = np.transpose(mat)
    lower = matT[colNum:, :]
    upper = matT[0:colNum, :]
    newMatT = np.array([upper, col, lower])
    newMat = np.transpose(newMatT)
    
    return newMat
    
def choose_middle(arr):
    """
    Returns the middle element of an array.
    """
    if len(arr) == 0:
        print('WARNING: array ' + str(arr) + ' is empty. Returning empty array.')
        return []
    
    return arr[int(len(arr)/2)]

def combine_repetitions(x, y, sigma):
    """
    Given x, y values with uncertainties in the y-value sigma, combines
    repeated measurements of y for the same x and computes the weighted 
    uncertainty.
    INPUTS
        x : array of x-values of data
        y : array of y-values of data
        sigma : array of uncertainty in y-values of data
    RETURNS
        xUnique : array of x-values for combined data
        yUnique : array of y-values for combined data
        sigmaUnique : array of weighted uncertainties in combined y-values of data
    """
    xUnique, inds = np.unique(x, return_index=True)
    # number of unique data points
    nUnique = len(inds)
    # initialize arrays to store combined y-values and uncertainties
    yUnique = np.zeros([nUnique])
    sigmaUnique = np.zeros([nUnique])
    # add length of array as a stopping point
    bins = np.append(inds, len(x))
    for i in range(nUnique):
        j1 = bins[i]
        j2 = bins[i+1]
        yUnique[i] = np.sum(y[j1:j2]/sigma[j1:j2]**2)/np.sum(1/sigma[j1:j2]**2)
        sigmaUnique[i] = np.sqrt(1/np.sum(1/sigma[j1:j2]**2))
    
    return xUnique, yUnique, sigmaUnique
    
    
def convert_condition(condition):
    """
    Converts old format of conditions
    ['Dry','Water', '3mM SDS', '6.5mM SDS', '10mM SDS', 'Water with 10mM SDS']
    to the new format
    ['Dry', 'Water', 'SDS 3.0 mM', 'SDS 6.5 mM', 'SDS 10 mM', 'Water - SDS 10 mM Rinse']
    and vice-versa.
    """
    oldConditionList = np.array(['Dry','Water', '3mM SDS', '6.5mM SDS',
                                 '10mM SDS', 'Water with 10mM SDS'])
    newConditionList = np.array(['Dry', 'Water', 'SDS 3.0 mM', 'SDS 6.5 mM',
                                 'SDS 10 mM', 'Water - SDS 10 mM Rinse'])
    index1 = np.where(oldConditionList == condition)[0]
    if len(index1) > 0:
        return newConditionList[index1[0]]
    else:
        index2 = np.where(newConditionList == condition)[0]
        return oldConditionList[index2[0]]
        
  
def double_gaussian( x, params ):
    (c1, mu1, sigma1, c2, mu2, sigma2) = params
    res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
          + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    return res

def double_gaussian_fit(x, y, params ):
    fit = double_gaussian( x, params )
    return (fit - y)


def find_trial(fileName):
    """
    Returns the number of the trial of the given video based on the file name.
    trial = -1 if it is to be ignored.
    """
    fpsInd = fileName.find('FPS_')
    trialInd = fpsInd + len('FPS_')
    lastInd = fileName.find('_c')
    if trialInd < lastInd:
        trialStr = fileName[trialInd:lastInd]
        if trialStr == 'X':
            trial = -1 # skip this trial
        else:
            try:
                trial = int(trialStr) - 1 # subtract 1 for proper 0-indexing
            except:
                trial = -1
    else: # trial was not marked, so it was the first trial
        trial = 0
        
    return trial

def fit_circle(x,y,center_estimate=(0,0)):
    """
    Fit the x and y points to a circle. Returns the circle's radius, center,
    and residue (a measure of error)
    """
    def calc_R(center):
        """
        Calculate the distance of each 2D point from the center (xc, yc) 
        """
        xc = center[0]
        yc = center[1]
        return np.sqrt((x-xc)**2 + (y-yc)**2)
    
    def f_2(center):
        """
        Calculate the algebraic distance between the data points and the mean
        circle centered at (xc, yc)
        """
        Ri = calc_R(center)
        return Ri - Ri.mean()

    center, ier = optimize.leastsq(f_2,center_estimate)
    
    Ri = calc_R(center)
    R = np.mean(Ri)
    residue   = sum((Ri - R)**2)
    return R, center, residue
    
def film_thickness(scaling, RPM, Q, condition, rCrit, data):
    """
    Estimates the thickness of the film [cm] produced by the impinging jet using
    one of the following four scalings:
        (1)
        (2)
        (3)
        (4)
    """
    # Derived Quantities and Constants
    jetRad = 0.2 # cm
    mu = 0.001 # viscosity [Pa*s]
    rho = 1000 # density [kg/m^3]
    nu = mu / rho # kinematic viscosity [m^2/s]
    g = 9.8 # gravitational acceleration [m/s^2]
    gamma_H2O = 0.0728 # surface tension of water [N/m]
    # Conversion factors
    m3PermL = 1E-6
    minPerSec = 1.0/60.0
    mPercm = 1.0/100.0
    radPerRotation = 2*np.pi # radians in one rotation
    
    # Converted quantities
    Q_m3_s = Q*m3PermL*minPerSec # flow rate in m^3/s
    jetRad_m = jetRad*mPercm # radius of impinging jet, [cm]
    RPM_rad_s = radPerRotation * minPerSec * RPM
    
    # initialize result
    filmThickness = 0
    # thickness = volume dispensed / wetted area
    if scaling == 'volumeDispensed':
        # Parse experiment data
        fps = data['fps']
        t0 = data['t0']
        time = data['time']
        aMax = data['aMax']
        aMin2 = data['aMin2']
        aMean = data['aMean']
        eP = data['excessPerimeter']
        indCrit = [i for i in range(len(aMean)-1) if (aMean[i+1] >= rCrit and aMean[i] <= rCrit)][0]
        tCrit = time[indCrit]
        tFlow = tCrit - t0
        volDispensed_m3 = Q_m3_s * tFlow # m^3
        wettedArea_m2 = np.pi * (mPercm * aMean[indCrit])**2 # m^2
        filmThicknessEst_m = volDispensed_m3 / wettedArea_m2 # m
        filmThickness = filmThicknessEst_m / mPercm # cm
    # Aristoff scaling (Bush and Aristoff, 2003, JFM, p. 233)
    elif scaling == 'aristoff':
        Re = Q_m3_s * jetRad_m / nu # dimensionless (nu in m^2/s)
        filmThickness = jetRad / (0.63 * Re**(1.0/3)) # [cm]
    # Assumes thickness of film is roughly the capillary length (2.7 mm for H2O)
    elif scaling == 'capillaryLength':
        filmThickness = np.sqrt(gamma_H2O / (rho * g)) / mPercm #update to consider SDS solutions?
    elif scaling == 'togashi':
        if RPM_rad_s == 0:
            filmThickness = 0 # mark as 0 to avoid division by 0 for stationary disk
        else:
            filmThickness_m = (6.0 * nu * Q_m3_s / (np.pi * (RPM_rad_s**2.0) * (2.0*jetRad_m)**2.0))**(1.0/3) # [m]
            filmThickness = filmThickness_m / mPercm # [cm]
    return filmThickness # cm


def gaussian_intersections(params):
    """
    Returns the intersection points of two gaussians, given by their parameters
    """
    # tolerance for imaginary component
    tol = 1E-10
    # magnitude, mean, stdev of first and second gaussians
    (a1, mu1, s1, a2, mu2, s2) = params
    # if negative amplitude or mean value outside uint8 range, no meaningful intersections
    if a1 <= 0 or a2 <= 0 or mu1 < 0 or mu1 > 255 or mu2 < 0 or mu2 > 255:
        return []
    # coefficients of quadratic equation whose solution gives intersection points
    a = -1/(2.0*s1**2) + 1/(2.0*s2**2)
    b = mu1/s1**2 - mu2/s2**2
    c = np.log(a1/a2) - mu1**2/(2.0*s1**2) + mu2**2/(2.0*s2**2)
    # solutions of quadratic equation
    x1 = (-b + np.sqrt(b**2 - 4*a*c))/(2.0*a)
    x2 = (-b - np.sqrt(b**2 - 4*a*c))/(2.0*a)
    # if result is complex, no intersection points
    if np.abs(np.imag(x1)) > tol:
        return []
    # otherwise remove residual imaginary components
    else:
        return np.real(x1), np.real(x2)
    
    
def generate_circle(R,center,N=100,t0=0.0,t1=2.0*np.pi):
    """
    Generate an array of x and y values that lie evenly spaced on a circle 
    with the specified center and radius.
    """
    theta = np.linspace(t0,t0+t1,N)
    y = R*np.sin(theta) + center[1]
    x = R*np.cos(theta) + center[0]
    return x,y
    
def generate_rectangle(xVert, yVert):
    """
    Generate an array of x and y values that outline a rectangle defined by the
    two opposing vertices given in xVert and yVert
    xVert, yVert: each is a numpy array of two values, a min and a max, not 
    necessarily in that order
    """
    # extract mins and maxes
    xMin = int(min(xVert))
    xMax = int(max(xVert))
    yMin = int(min(yVert))
    yMax = int(max(yVert))
    # initialize list of values
    X = []
    Y = []
    # top border
    for x in range(xMin,xMax):
        X += [x]
        Y += [yMin]
    # right border
    for y in range(yMin, yMax):
        X += [xMax]
        Y += [y]
    # bottom border
    for x in range(xMax, xMin, -1):
        X += [x]
        Y += [yMax]
    # left border
    for y in range(yMax, yMin, -1):
        X += [xMin]
        Y += [y]
    return np.array(X), np.array(Y)
    
def fit_ellipse(x,y):
    """
    Fit the x and y points to an ellipse. Returns the radii, center,
    and and angle of rotation. Taken directly from:
        http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    """
    
    def fit(x,y):
        x = x[:,np.newaxis]
        y = y[:,np.newaxis]
        D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T,D)
        C = np.zeros([6,6])
        C[0,2] = C[2,0] = 2; C[1,1] = -1
        E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
        n = np.argmax(np.abs(E))
        a = V[:,n]
        return a
        
    def ellipse_center(a):
        b,c,d,f,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[0]
        num = b*b-a*c
        x0=(c*d-b*f)/num
        y0=(a*f-b*d)/num
        return np.array([x0,y0])
    
    def ellipse_angle_of_rotation(a):
        b,c,a = a[1]/2, a[2], a[0]
        return 0.5*np.arctan(2*b/(a-c))
    
    def ellipse_axis_length(a):
        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
        down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        res1=np.sqrt(up/down1)
        res2=np.sqrt(up/down2)
        return np.array([res1, res2])
    
    a = fit(x,y)
    center = ellipse_center(a)
    theta = ellipse_angle_of_rotation(a)
    [R1,R2] = ellipse_axis_length(a)

    return R1, R2, center, theta
    
def generate_ellipse(R1,R2,center,theta,N=100):
    """
    Generate an array of x and y values that lie on an ellipse with the 
    specified center, radii, and angle of rotation (theta)
    """
    t = np.linspace(0.0,2.0*np.pi,N)
    x = R1*np.cos(t)*np.cos(theta) - R2*np.sin(t)*np.sin(theta) + center[0]
    y = R1*np.cos(t)*np.sin(theta) + R2*np.sin(t)*np.cos(theta) + center[1]
    return x,y
    

def get_file_name_from_path(filePath):
    """
    Returns just the file name, no folder headings.
    """
    indsBackslash = find_substring('\\',filePath)
    # check if there are any folder headings--if not return filepath
    if not indsBackslash:
        fileName = filePath
    else:
        # get index of last occurrence of backslash
        iLast = indsBackslash[-1]
        # extract characters 2 after last '\\' (1 to account for length of '\\')
        fileName = filePath[iLast+1:]
    
    return fileName
 
def find_last(substr, string):
    """
    Returns the index of the last occurrence of the substring.
    """
    inds = find_substring(substr, string)
    return inds[-1]

def find_substring(substr,string):
    """
    Finds the indices for given substring.
    """
    inds = []
    i = string.find(substr)
    while i != -1:
        inds += [i]
        # find first occurrence of substring starting at index i+1
        i = string.find(substr, i+1)
    
    return inds

def plot_circles(R,center,N=15):
    """
    Plot a set of concentric circles on an existing plot where the outermost
    circle is defined by the specified radius and center.
    """
    Ri = np.linspace(0,R,N+1)[1:]
    color = ['r--','b--','m--','y--']
    for i in range(N):
        xi, yi = generate_circle(Ri[i],center)
        j = i % 4
        plt.plot(xi,yi,color[j],alpha=0.3)
        
def plt_show_image(image):
    """
    This removes tick marks and numbers from the axes of the image and fills 
    up the figure window so the image is easier to see.
    """
    plt.imshow(image)
    plt.axis('off')
    plt.axis('image')
    plt.tight_layout(pad=0)
    
def generate_polygon(x,y,N):
    """
    Generate an array of x and y values that lie evenly spaced along a polygon
    defined by the x and y values where it is assumed that the first value
    is also the last value to close the polygon
    """
    # Add the first point to the end of the list and convert to array if needed
    if type(x) == list:
        x = np.array(x + [x[0]])
        y = np.array(y + [y[0]])
    else:
        x = np.append(x,x[0])
        y = np.append(y,y[0])
        
    # Parameterize the arrays and interpolate
    d = [get_distance((x[i],y[i]),(x[i+1],y[i+1])) for i in range(len(x)-1)]
    d = np.cumsum([0]+d)
    t = np.linspace(0,d[-1],N)
    fx = interp1d(d,x)
    fy = interp1d(d,y)
    x = fx(t)
    y = fy(t)
    
    return x,y
    
#def polygon_perimeter(pts):
#    """
#    Calculates the perimeter of the polygon defined by the tuples in the array
#    pts
#    """
#    # Calculate length of edge from last point to first point
#    perimeter = 0
#    d = get_distance(pts[len(pts)-1],pts[0])
#    perimeter += d
#    # Calculate length of remaining edges
#    for i in range(0,len(pts)-1):
#        d = get_distance(pts[i],pts[i+1])
#        perimeter += d
#    return perimeter
    
def generate_periodic_circle(R, var, center, freq, fn):
    """
    Generates points that map out a circle with a given periodic function
    overlayed.
    
    fn can be the following:
    Fun.sine -> overlays sine wave with amplitude "var"
    Fun.sawtooth -> overlays the increasing sawtooth that begins at (R-var) and
    increases up to (R+var) in each period
    Fun.triangle -> overlays the increasing-decreasing triangle, which begins at
    (R-var), reaches (R+var) halfway through the period, and then decreases
    back to (R-var) by the end of the period.
    """
    xCenter = center[0]
    yCenter = center[1]
    # The perimeter for a triangle wave is about sqrt((circum)^2+(2*var*freq)^2),
    # A little bit is added to get the formula used below, just in case.
    nPts = int(2*np.sqrt((2*np.pi*R)**2.0 + (2*var*freq)**2.0)) # ensure sufficient points to fill perimeter
    theta = np.linspace(0,2*np.pi, nPts)
    # Initalize xy, array of tuples of x- and y-values
    xy = np.array([[0,0]])
    if freq == 0:
        period = 0
    else:
        period = 2*np.pi/freq
    for i in range(len(theta)):
        pert = fn(theta[i], period, var)
        r = R + pert
        # Convert from polar to Cartesian coordinates
        x = r*np.cos(theta[i]) + xCenter
        y = r*np.sin(theta[i]) + yCenter
        xy = np.concatenate((xy,np.array([[x,y]])))
    return xy[1:len(xy)]
    
def sawtooth(x, period, var):
    """
    Calculates y-value for an increasing sawtooth with amplitude "var" above
    and below y = 0, i.e. it begins at (0,-var), passes through (period/2,0),
    and ends at (period,+var) in each period. 
    """
    if period == 0:
        return x
    else:
        x = x % period
        frac = x/period
        y = (2*frac-1)*var
        return y
    
def sine(x, period, var):
    """
    Calculates y-value for a sine wave starting at the origin.
    """
    return var*np.sin((2*np.pi/period)*x)
    
def triangle(x, period, var):
    """
    Calculates y-value for an increasing-decreasing triangle function with 
    amplitude "var" above and below y = 0, i.e. it begins at (0,-var), 
    increases to (period/2,+var), and decreases to (period,-var).
    """
    if period == 0:
        return x
    else:
        x = x % period
        frac = 2*abs(0.5 - (x/period))
        y = (1-2*frac)*var
        return y
        
def get_angle(pt1,pt2,pt3):
    """
    Uses law of cosines to calculate angle defined by given points, where pt2
    is the vertex of the angle
    """
    a = float(get_distance(pt1,pt2))
    b = float(get_distance(pt2,pt3))
    c = float(get_distance(pt1,pt3))
    angle = np.arccos((a**2 + b**2 - c**2)/(2*a*b)) # Law of Cosines 
    
    return angle
    
def get_corrected_arclength(pts,closed=False):
    """
    Smooths the digital curve defined by the row-column tuples in the numpy
    array "pts" using a 5-point average, i.e., it replaces each tuple with an 
    average of the two previous points, the two succeeding points, and the 
    point itself. The arc-length is then calculated by scaling up the image
    to 3-decimal-place precision, applying the OpenCV arcLength function, and
    scaling back down.
    """
    
    l = len(pts)
    ptsDown2 = np.concatenate((pts[2:l],pts[0:2]))
    ptsDown1 = np.concatenate((pts[1:l],np.array([(pts[0][0],pts[0][1])])))
    ptsUp1 = np.concatenate((np.array([(pts[l-1][0],pts[l-1][1])]),pts[0:l-1]))
    ptsUp2 = np.concatenate((pts[l-2:l],pts[0:l-2]))
    summedPts = ptsDown2 + ptsDown1 + pts + pts + ptsUp1 + ptsUp2
    avePts = summedPts/5.0
    zoomAvePts = np.round(avePts)
    arcLength = cv2.arcLength(zoomAvePts.astype(int),closed)
    
    return arcLength
    
def get_distance(pt1,pt2):
    """
    Calculates distance between two points given as tuples of row and col
    """
    x1 = pt1[1]
    y1 = pt1[0]
    x2 = pt2[1]
    y2 = pt2[0]
    d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return d

def get_radial_distance(p1,p2,R):
    """ Get the distance between 2 points, but call the distance zero if it is
    less than a tenth of the radius (area where flow is blocked by jet arm head).
    INPUT:
        p1 = tuple of (x,y) coordinates
        p1 = tuple of (x,y) coordinates
    OUTPUT:
        d = distance between points
    """
    d = np.sqrt((p1[0]-p2[0])**2.0 + (p1[1]-p2[1])**2.0)
    if d < 0.1*R:
        d = 0.0
    elif d > R:
        d = R
    return d
        
def get_theta(p1,p2):
    """
    Compute the angle of the point on the disk relative to the center where
    p2 is the center of the disk.
    """
    
    dy = p1[1] - p2[1]
    dx = p1[0] - p2[0]
    theta = atan2(dy,dx)
    return theta    
    
def get_linear_distance(p1,p2):
    """ Get the distance between 2 points, but call the distance zero if the 
    first point is to the left of the second point.
    INPUT:
        p1 = tuple of (x,y) coordinates
        p1 = tuple of (x,y) coordinates
    OUTPUT:
        d = distance between points
    """
    if p1[0] < p2[0]:
        d = 0
    else:
        d = np.sqrt((p1[0]-p2[0])**2.0 + (p1[1]-p2[1])**2.0)
    return d
    
def convert_flowrate(setting):
    """
    Return the actual flow rate using the flow rate setting and the conversion
    from the pump calibration work.
    The setting and the returned flow rate are in units of mL/min.
    """
    return 0.8991*setting - 62.339
    
def rotate_points(x,y,theta,center=[0,0],units='radians'):
    """
    Take a set of points and rotate them and angle theta about a center point.
    Theta in degrees.
    """
    # Convert theta to radians
    if units == 'degrees':
        theta = theta/180.0*np.pi
    elif units == 'radians':
        pass
    else:
        print('Invalid input parameter for angle units! Assuming radians')
    
    # Compute rotation matrix constants
    s = np.sin(theta)
    c = np.cos(theta)

    # translate points so that the center is at the to origin:
    x -= center[0]
    y -= center[1]

    # rotate points by multiplying by rotation matrix
    xnew = x*c - y*s
    ynew = x*s + y*c

    # translate points back to original location
    xnew += center[0]
    ynew += center[1]
  
    return xnew,ynew
 
def get_flow_rates(fileName, hdr, innerConversion=1.0, outerConversion=1.0):
    """Extracts just the flowrates of the inner and outer streams from the file
    name. The header (hdr) must include all text up to and including the "_" 
    before the flow rates
    """
    # get length of header for appropriate offset
    l = len(hdr)
    # index for first character after header
    i = fileName.find(hdr) + l
    # strings for outer and inner flowrates based on naming convention
    outerStr = fileName[i:i+4]
    iDash = fileName.find('-')
    # dash present indicates decimal point in inner flow rate
    if iDash!=-1:
        innerConversion = 1/100.0
        innerStr = fileName[iDash+1:iDash+3]
        innerFlowRate = actual_flow_rate(int(innerStr), innerConversion)
    else:
        innerStr = fileName[i+5:i+9]
        innerFlowRate = actual_flow_rate(int(innerStr), innerConversion) # ul/min
    outerFlowRate = actual_flow_rate(int(outerStr), outerConversion) # ul/min
    return innerFlowRate, outerFlowRate
    
 
def get_crossings(arr, thresh):
    """
    returns numpy array with True at each index where the input array arr crosses
    the given value thresh
    """
    belowThresh = arr < thresh
    crossings = np.logical_xor(belowThresh,np.roll(belowThresh,-1))
    return crossings


def get_save_name(prefix, RPM, Q, cond, ext):
    # prefix should include save folder plus beginning of save name
    # Create save name
    saveName = ''
    if prefix != '':
        saveName += prefix + "_"
    if RPM != '':
        saveName += "RPM_%d_" % (RPM)
    if Q != '':
        saveName += "Q_%d_" % (int(round(Q,-1)))
    if cond != "":
        saveName += "cond_"
        if cond.find('SDS') < 0:
            saveName += cond
        elif cond.find('Water') < 0:
            saveName += 'SDS_'
            concentration = float(cond[0:cond.find('mM')])
            saveName += "%d" % int(10*concentration)
        else:
            saveName += 'SDS_100_on_Water'
        saveName += "_"
        
    saveName = saveName[:len(saveName)-1] + ext
        
    return saveName
    
def interp_zeros(mat):
    """
    Interpolates the zero-valued elements of a matrix using the surrounding
    elements. Matrix "mat" should be a numpy array.
    """
    [m, n] = mat.shape
    for r in range(m):
        for c in range(n):
            if mat[r, c] == 0:
                count = 0
                for i in range(max(0,r-1), min(r+1,m)):
                    for j in range(max(0,c-1), min(c+1,n)):
                        mat[r, c] += mat[i,j]        
                        count += 1
                mat[r,c] /= count        
    return mat
 

def longest_sequence(crossings, indicator=True):
    """
    Returns the index and length of the longest sequence of values between
    crossings (crossings indicated by indicator, non-crossings indicated by 
    any other value).
    Essentially returns length of longest sequence of non-indicator values
    and the index of the indicator value preceding it.
    """
    iIndicators = np.where(crossings==indicator)[0]
    # length of sequences of between indicator values, -1 to exclude indicator value
    seqLengths = np.diff(iIndicators)-1
    # index of indicator preceding longest sequence of non-indicator values
    iSeq = iIndicators[np.argmax(seqLengths)]
    # length of longest sequence
    lengthSeq = np.max(seqLengths)
    
    return iSeq, lengthSeq

def overlapping_gaussians(params, nSigma=2.0):
    """
    Returns True if Gaussians are overlapping, false if not
    """
    (a1, mu1, s1, a2, mu2, s2) = params
    muDiff = np.abs(mu1-mu2)
    return bool(muDiff < nSigma*max(s1,s2))


def power_law_fit(x, y, sigma):
    """
    Computes a power-law fit of given (x,y) points and uncertainties sigma, 
    returning the coefficients and uncertainities of the fit.
    """
    xlog = np.log(x)
    ylog = np.log(y)
    sigmaYLog = np.divide(sigma, y)
    p, V = np.polyfit(xlog, ylog, 1, w=1/sigmaYLog, cov=True)
    m, b = p
    sigmaM = np.sqrt(V[0,0])
    sigmaB = np.sqrt(V[1,1])
    A = np.exp(b)
    sigmaA = A*sigmaB
    
    return m, A, sigmaM, sigmaA

def stream_width(innerFlowrate, outerFlowrate, innerDiameter, innerMu=1E-3, outerMu=1E-3):
    """
    Gives expected radius of inner stream inside a cylindrical tube of a given
    inner diameter given the inner and outer flowrates. Included effect of
    viscosity on Aug 13, 2018 based on the results from sheath_flow_diameter.nb
    and the viscosities are set to be equal to that of water by default.
    """
    if innerMu != outerMu:
        print('WARNING: Formula is incorrect for unequal viscosities.')
    # previous code
#    num = innerFlowrate*innerMu
#    den = (innerFlowrate + outerFlowrate)*innerMu + np.sqrt(outerFlowrate*innerMu*\
#          (outerFlowrate*innerMu + innerFlowrate*outerMu))
#    streamWidth = innerDiameter*np.sqrt(num/den)
    streamWidth = innerDiameter*np.sqrt(innerFlowrate/(2*outerFlowrate))
    
    return streamWidth
    
    
if __name__ == '__main__':
    plt.close('all')
    R = 1
    center = [0,0]
    x,y = generate_circle(R,center,7,t0=0,t1=np.pi/2)
    x1,y1 = rotate_points(x,y,-90,center,'degrees')
    plt.plot(x,y,'ro',x1,y1,'b.')
    plt.axis('equal')
