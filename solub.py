# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:38:57 2019

@author: Andy
"""

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.optimize import root



def interp_prop(p, T, p_arr_list, prop_arr_list, T_list, dp=10, fit_quadratic=False,
                s_p=0, s_T=0):
    """
    Interpolates measurements of a property to new pressures and temperatures by
    first interpolating to a fixed spacing of pressures within a given temperature
    and then interpolating along the temperature at one of the interpolated
    values of pressure.
    PARAMETERS:
        p_arr_list : list of numpy arrays of varying lengths
            Each list should start with 0 pressure [kPa].
    """
    # starting value of minimal max pressure [kPa]--should be very high
    p_min_max = 1E4
    M = len(p_arr_list)
    for i in range(M):
        p_min_max = min(np.max(p_arr_list[i]), p_min_max)
    p_interp = np.arange(0, p_min_max, dp)
    # dimensions of interpolated matrix
    N = len(p_interp)
    # initialize list of arrays of interpolated solubilities
    prop_interp_mat = np.zeros([M, N])
    # interpolate solubilities at each pressure
    for i in range(M):
        p_arr = p_arr_list[i]
        prop_arr = prop_arr_list[i]
        # remove nans--second argument must be a list
        p_arr, prop_arr = remove_nan_entries(p_arr, [prop_arr])
        inds_sort = np.argsort(p_arr)
        p_arr = p_arr[inds_sort]
        prop_arr = prop_arr[inds_sort]
        prop_interp_mat[i,:] = np.interp(p_interp, p_arr, prop_arr)
    # identify index of interpolated pressure closest to desired pressure
    i_p = np.argmin(np.abs(p_interp-p))
    # interpolate solubilities at desired pressure for desired temperature
    T_arr = np.array(T_list)
    i_below = np.where(T>T_arr)[0][0]
    prop = np.interp(T, T_arr[i_below:i_below+2], prop_interp_mat[i_below:i_below+2, i_p])
    # quadratic fit of temperature effect on solubility
    if fit_quadratic:
        coeffs = np.polyfit(T_arr, prop_interp_mat[:, i_p], 2)
        prop = np.polyval(coeffs, T)

    # compute errors--first consider errors due to uncertainty in pressure
    if s_p > 0:
        s_p_prop = (interp_prop(p+s_p, T, p_arr_list, prop_arr_list, T_list,
                                fit_quadratic=fit_quadratic) - \
                    interp_prop(p-s_p, T, p_arr_list, prop_arr_list, T_list,
                                            fit_quadratic=fit_quadratic))/2
    else:
        s_p_prop = 0
    # next consider errors due to uncertainty in temperature
    if s_T > 0:
        s_T_prop = (interp_prop(p, T+s_T, p_arr_list, prop_arr_list, T_list,
                                fit_quadratic=fit_quadratic) - \
                    interp_prop(p, T-s_T, p_arr_list, prop_arr_list, T_list,
                                            fit_quadratic=fit_quadratic))/2
    else:
        s_T_prop = 0
    # add errors in quadrature to get overall uncertainty
    s_prop = np.sqrt(s_p_prop**2 + s_T_prop**2)
    # only return error if non-zero
    if s_prop == 0:
        return prop
    else:
        return prop, s_prop


def m2p_co2_polyol(m_co2, m_polyol, rho_polyol=1.084, V=240, p0=30E5,
                   polyol_name='VORANOL 360'):
    """
    Converts the mass of carbon dioxide (dry ice) in the Parr reactor to the
    expected pressure based on solubility in the polyol.

    m_co2 : mass of dry ice upon sealing Parr reactor [g]
    m_polyol : mass of polyol [g]
    rho_polyol : density of polyol, VORANOL 360 [g/mL]
    V : available internal volume of Parr reactor [mL]
    p0 : initial guess of pressure for nonlinear solver (~expected pressure) [Pa]

    returns :
        pressure expected based on mass of CO2 [Pa]
    """
    # volume of polyol [mL]
    V_polyol = m_polyol / rho_polyol
    # volume of gaseous head space [mL]
    V_gas = V - V_polyol

    # interpolation functions
    f_sol = interpolate_dow_solubility(polyol_name)
    f_rho = rho_co2()

    # equation to solve for pressure
    def fun(p, m_co2=m_co2, f_rho=f_rho, V_gas=V_gas, m_polyol=m_polyol,
            f_sol=f_sol):
        """
        Function to solve to determine pressure.
        """
        return m_co2 - f_rho(p)*V_gas - m_polyol*f_sol(p)

    result = root(fun, p0)
    p = result.x

    print('Mass in gas phase = %.2f g.' % (f_rho(p)*V_gas))
    print('Mass in liquid phase = %.2f g.' % (f_sol(p)*m_polyol))

    return p


def p2m_co2_polyol(p, m_polyol, rho_polyol=1.084, polyol_name='VORANOL 360',
                   V=240, m0=20):
    """
    Computes the required mass of dry ice (CO2) to add to a Parr reactor of
    volume V with a given mass of polyol to reach the desired pressure based on
    the solubility data of CO2 in this polyol.

    Parameters:
        p : int
            Desired final pressure in Parr reactor [bar]
        m_polyol : int
            Mass of polyol [g]
        rho_polyol : int, default 1.084
            Density of polyol [g/mL]
        polyol_name : string, default 'VORANOL 360'
            Name of the polyol (for loading corresponding solubility data)
        V : int, default 240
            Volume of Parr reactor [mL]
        m0 : int, default 20
            Initial guess for dry ice mass [g]

    Returns:
        m : int
            Mass of dry ice required to achieve desired pressure [g]
    """
    # volume of polyol [mL]
    V_polyol = m_polyol / rho_polyol
    # volume of gaseous head space [mL]
    V_co2_gas = V - V_polyol
    # Interpolate density of CO2 at given pressure [g/mL] and 25 C
    rho_co2_var = rho_co2(p=100*p)
    # Mass of CO2 in gas phase [g]
    m_co2_gas = rho_co2_var * V_co2_gas
    # Interpolate weight fraction of CO2 based on Dow solubility data
    w_co2 = interpolate_dow_solubility(polyol_name, p=p)
    # Calculate mass of CO2 dissolved in polyol [g]
    m_co2_dissolved = w_co2 * m_polyol
    # Total mass [g]
    m = m_co2_gas + m_co2_dissolved

    print('Mass in gas phase = %.2f g.' % (m_co2_gas))
    print('Mass in liquid phase = %.2f g.' % (m_co2_dissolved))

    return m


def interpolate_dow_solubility(polyol_name='VORANOL 360', p=None):
    """
    Returns an interpolation function for the solubilty of VORANOL 360 at 25 C
    in terms of weight fraction as a function of the pressure in bar.
    Will perform the interpolation if an input pressure p is given.

    Parameters:
        polyol_name : string, default 'VORANOL 360'
            Name of polyol, used to load corresponding solubility data
        p : int, default None
            Pressure at which solubility is desired to be interpolated [bar]

    Returns:
        if pressure 'p' not provided (p=None, default):
            f_sol : interpolation function
                Function to interpolate weight fraction solubility given pressure in bar
        else if pressure 'p' provided:
            f_sol(p) : int
                Weight fraction solubility of CO2 in polyol at pressure p [bar]
    """
    # constants
    psi2pa = 1E5/14.5

    # copy-paste data from file "co2_solubility_pressures.xlsx" for VORANOL 360
    if polyol_name == 'VORANOL 360':
        data = np.array([[0,0],
            [198.1, 0.0372],
            [405.6, 0.0821],
            [606.1, 0.1351],
            [806.8, 0.1993],
            [893.9, 0.2336]])
    elif polyol_name == 'VORANOL 2110B':
        # left column: pressure in psi; right column: solubility in [w/w]
        data = np.array([[0, 0],
                    [193.5, 0.0509],
                    [388.7, 0.1088],
                    [614.8, 0.1865],
                    [795.1, 0.2711],
                    [881.5, 0.3273]])
        # Naples data at 30.5 C
        # data = np.array([[0, 0],
        #     [500.046, 0.014721248],
        #     [994.929, 0.030183997],
        #     [1489.9723, 0.046353253],
        #     [1982.3343, 0.063205735],
        #     [2472.1965, 0.080613988],
        #     [2959.656, 0.098423283],
        #     [3445.9158, 0.117802366],
        #     [3930.2006, 0.137732216],
        #     [4415.434, 0.158874026],
        #     [4898.8536, 0.18212937],
        #     [5289.94, 0.201551396],
        #     [4566.438, 0.166379887],
        #     [3837.4145, 0.1341855],
        #     [3103.4211, 0.104465493],
        #     [2365.2733, 0.077000957],
        #     [1622.265, 0.05120924],
        #     [877.9971, 0.026891273],
        #     [680.0009, 0.020519271],
        #     [481.7703, 0.014287419],
        #     [283.2508, 0.007999299],
        #     [233.8055, 0.006332145],
        #     [183.4303, 0.004722989],
        #     [129.8668, 0.003064421],
        #     [79.9356, 0.00142699],
        #     # extrapolated value from crude approximation of slope
        #     [6000, 0.24]])
        # # convert pressures to psi
        # data[:,0] = data[:,0]*14.5E-2
    else:
        print("Data for polyol not found. Ending calculation.")
        return
    # make sure data are sorted by Pressure
    inds_sorted = np.argsort(data[:,0])
    data = data[inds_sorted,:]
    # first column is pressure in psia
    p_data_psia = data[:,0]
    # second column is solubility in fraction w/w
    solubility_data = data[:,1]
    # convert pressure from psi to Pa
    p_data_pa = psi2pa * p_data_psia
    # define interpolation function
    f_sol = interp1d(p_data_pa, solubility_data, kind="cubic")

    # Return weight fraction if pressure p provided
    if p:
        # convert pressure to Pascals
        p *= 1E5
        # if the pressure is above the range of the data, extrapolate
        if p > np.max(p_data_pa):
            a, b = np.polyfit(p_data_pa, solubility_data, 1)
            solubility = a*p + b
        else:
            solubility = f_sol(p)
        return solubility
    # Otherwise, return interpolation function for weight fraction vs. pressure
    else:
        return f_sol


def rho_co2(p=None, T=25, eos_file_hdr='eos_co2_', ext='.csv'):
    """
    Returns an interpolation function for the density of carbon dioxide
    according to the equation of state (data taken from
    webbook.nist.gov.
    The density is returned in term of g/mL as a function of pressure in kPa.
    Will perform the interpolation if an input pressure p is given.
    PARAMETERS:
        p : int (or array of ints)
            pressure in kPa of CO2
        T : float
            temperature in Celsius (only to one decimal place)
        eos_file_hdr : string
            File header for equation of state data table
    RETURNS:
        rho : same type as p
            density in g/mL of co2 @ given temperature
    """
    # get decimal and integer parts of the temperature
    dec, integ = np.modf(T)
    # create identifier string for temperature
    T_tag = '%d-%dC' % (integ, 10*dec)
    # dataframe of appropriate equation of state (eos) data from NIST
    df_eos = pd.read_csv(eos_file_hdr + T_tag + ext, header=0)
    # get list of pressures of all data points [kPa]
    p_co2_kpa = df_eos['Pressure (kPa)'].to_numpy(dtype=float)
    # get corresponding densities of CO2 [g/mL]
    rho_co2_var = df_eos['Density (g/ml)'].to_numpy(dtype=float)
    # remove repeated entries
    p_co2_kpa, inds_uniq = np.unique(p_co2_kpa, return_index=True)
    rho_co2_var = rho_co2_var[inds_uniq]
    # create interpolation function and interpolate density [g/mL]
    f_rho = interp1d(p_co2_kpa, rho_co2_var)
    rho = f_rho(p)

    return rho


def interpolate_rho_co2(p=None):
    """
    Returns an interpolation function for the density of carbon dioxide
    according to the equation of state (data taken from
    http://www.peacesoftware.de/einigewerte/co2_e.html) at 25 C.
    The density is returned in term of g/mL as a function of pressure in kPa.
    Will perform the interpolation if an input pressure p is given.

    OBSOLETE: use rho_co2() instead.
    """

    return rho_co2(p=p)

def interpolate_eos_co2(quantity, value=None):
    """
    Returns an interpolation function for the density of carbon dioxide
    according to the equation of state (return_p=False) (data taken from
    http://www.peacesoftware.de/einigewerte/co2_e.html) at 25 C or the pressure
    given the density (if a value is given).
    The density is returned in term of g/mL as a function of pressure in Pascals,
    and the pressure is returned in terms of Pascals given density in terms of
    g/mL.
    Will perform the interpolation if a given value is given.
    """
    #pressure in Pa
    p_co2_pa = 1E5*np.arange(0,75,5)
    # density in g/mL (at 25 C)
    rho_co2_var = np.array([0, 9.11, 18.725, 29.265, 39.805, 51.995, 64.185, 78.905,
                            93.625, 112.9625, 132.3, 151.9, 258.4, 737.5, 700.95])/1000

    # determine appropriate interpolation function
    if quantity=='p':
        f = interp1d(rho_co2_var, p_co2_pa, kind="cubic")
    elif quantity=='rho':
        f = interp1d(p_co2_pa, rho_co2_var, kind="cubic")
    else:
        print("please select a valid quantity: ''rho'' or ''p''")

    # Return the interpolation function
    if value==None:
        return f
    # Otherwise return the corresponding quantity given a value for the input
    else:
        return f(value)


def missing_mass(rho_V, rho_L, V_V, V_L, m):
    """Computes missing mass by comparing estimate of mass in liquid and vapor phases."""
    m_pred = rho_V*V_V + rho_L*V_L
    return m - m_pred


def solub_gc(p_cal, T_cal, V_hplis, frac_cal, peak_frac, spec_vol, s_peak_frac=0):
    """
    Converts measurement of peak area with gas chromatography (GC) into a
    measurement of gas solubility.
    """
    # compute the fraction of the calibration mass of 100% CO2 observed in experiment
    frac_co2 = frac_cal*peak_frac
    # density of CO2 under given conditions [g/mL]
    rho_co2_cal = rho_co2(p_cal, T_cal)
    # mass of CO2 in HPLIS groove [g]
    m_co2_hplis = frac_co2*rho_co2_cal*V_hplis

    # density of co2 in HPLIS groove (estimated to be same as polyol-CO2 mixture as estimated w/ Naples data) [g/mL]
    rho_co2_hplis = 1/spec_vol # changing to density of pure CO2 at given p and T has negligible effect on result
    # Volume of co2 in HPLIS [mL]
    V_co2_hplis = m_co2_hplis / rho_co2_hplis
    # Volume of polyol in HPLIS [mL]
    V_poly_hplis = V_hplis - V_co2_hplis
    # density of polyol in HPLIS [g/mL]--estimated to be same as polyol-CO2 mixture since doesn't change much with CO2
    rho_poly_hplis = 1/spec_vol

    # mass of polyol [g]
    m_poly_hplis = rho_poly_hplis*V_hplis
    # total mass in HPLIS [g]
    m_hplis = m_poly_hplis + m_co2_hplis
    # estimated solubility [w/w]
    solubility = m_co2_hplis / m_hplis

    # estimate uncertainty
    if s_peak_frac > 0:
        s_solubility = (solub_gc(p_cal, T_cal, V_hplis, frac_cal, \
                                peak_frac+s_peak_frac, spec_vol) - \
                    solub_gc(p_cal, T_cal, V_hplis, frac_cal, \
                                            peak_frac-s_peak_frac, spec_vol))/2
    else:
        s_solubility = 0

    if s_solubility == 0:
        return solubility
    else:
        return solubility, s_solubility


def remove_nan_entries(nan_containing_arr, accompanying_arr_list):
    """
    Removes entries all arrays in the accompanying_arr_list where the
    nan_containing_arr has a nan.
    PARAMETERS:
        nan_containing_arr : numpy array of floats/ints
            Array that user thinks might contain nans
        accompanying_arr_list : list of numpy arrays
            List of arrays of same length as nan_containing_arr. All entries
            with the same index as entries having nans in nan_containing_arr
            will be removed.
    RETURNS:
        nan_free_arr_list : list of numpy arrays
            List of nan_containing_arr and all elements of accompanying_arr_list,
            now with the entries for which nan_containing_arr had nans removed.
    """
    # number of elements of given array
    n = len(nan_containing_arr)
    # identify indices of entries that are not nans
    inds_not_nan = np.logical_not(np.isnan(nan_containing_arr))
    # initialize result with given array having nans removed
    nan_free_arr_list = [nan_containing_arr[inds_not_nan]]
    # loop through accomapanying arrays and add to result after removing same entries
    for arr in accompanying_arr_list:
        assert len(nan_containing_arr)==n, "Arrays not of the same length."
        nan_free_arr_list += [arr[inds_not_nan]]

    return nan_free_arr_list


def usb_2_actual_pressure(p_usb, lo_usb=-3, hi_usb=639, lo_cpu=1, hi_cpu=655, cpu2actual=1/0.895):
    """
    Converts pressure read during usb powering with Samsung 5V 2.0 A USB wall adapter to
    predicted actual pressure based on empirical measurements, particularly on pp. 45 and 51
    """
    p_cpu = (hi_cpu-lo_cpu)/(hi_usb-lo_usb)*p_usb + (lo_cpu-lo_usb)
    p_actual = cpu2actual * p_cpu

    return p_actual

def dc_2_actual_pressure(p_dc, p_lo=0, p_hi=1500, cts_lo=207, cts_hi=1023, cts_lo_actual=208, cts_hi_actual=914):
    """
    Converts pressure read during DC powering with
    predicted actual pressure based on empirical measurements based on calibration on pp. 64-65.
    """
    cts = (cts_hi-cts_lo)*(p_dc-p_lo)/(p_hi-p_lo) + cts_lo
    p_actual = (p_hi-p_lo)*(cts-cts_lo_actual)/(cts_hi_actual-cts_lo_actual) + p_lo

    return p_actual
