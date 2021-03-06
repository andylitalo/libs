"""
@polyco2.py contains functions used to query the thermophysical properties of
polyol-CO2 mixtures.

Most of the data come from experiments performed from June to August, 2019 in
the Di Maio lab at the University of Naples by Andy Ylitalo.

@author Andy Ylitalo
@date October 21, 2020
"""

# imports standard libraries
import numpy as np
import scipy.interpolate
import scipy.optimize
import pandas as pd

# imports custom libraries
import fn

# CONVERSIONS
s_2_ms = 1000
m_2_um = 1E6
kPa_2_Pa = 1000
gmL_2_kgm3 = 1000
cm2s_2_m2s = 1E-4


def calc_c_s(p, polyol_data_file):
    """
    Estimates the saturation concentration of CO2 in a polyol solution using
    interpolated measurements of solubility.

    Parameters
    ----------
    p : float
        pressure at which to estimate the saturation concentration [Pa]
    polyol_data_file : string
        name of file containing polyol data [.csv]

    Returns
    -------
    c_s : float
        concentration of CO2 in polyol-CO2 solution [kg/m^3] at the given pressure p
    """
    p_arr, c_s_arr = load_c_s_arr(polyol_data_file)
    # interpolates value to match the given pressure [kg CO2 / m^3 solution]
    c_s = np.interp(p, p_arr, c_s_arr)

    return c_s


def calc_D(p, polyol_data_file):
    """
    Estimates the diffusivity of CO2 in polyol at the given pressure
    by interpolating available measurements using G-ADSA. The two methods
    used to estimate the diffusivity, square-root fit of the initial transient
    and exponential fit of the final plateau are averaged to reduce the effects
    of noise/experimental error.

    If p is above the experimentally measured range, it is replaced with the
    maximum pressure for which there is a measurement. If below, it is replaced with
    the minimum. This is preferable since we are just trying to make some rough
    estimates as a demonstration of this method right now. More precise measurements
    in the future will require a more sophisticated approach).

    Parameters
    ----------
    p : float
        pressure at which to estimate the diffusivity [m^2/s]
    polyol_data_file : string
        name of file containing polyol data [.csv]

    Returns
    -------
    D : float
        diffusivity of CO2 in polyol [m^2/s] at the given pressure p
    """
    p_arr, D_sqrt_arr, D_exp_arr = load_D_arr(polyol_data_file)

    return calc_D_raw(p, p_arr, D_sqrt_arr, D_exp_arr)


def calc_D_raw(p, p_arr, D_sqrt_arr, D_exp_arr):
    """
    Calculates diffusivity as in calc_D as a function of pressure, but using
    raw arrays for interpolation.

    Parameters
    ----------
    p : float
        pressure at which to estimate the diffusivity [m^2/s]
    p_arr : N x 1 numpy array of floats
        array of pressures for interpolation [Pa]
    D_sqrt_arr : N x 1 numpy array of floats
        array of diffusivities [m^2/s] estimated with squareroot method
    D_exp_arr : N x 1 numpy array of floats
        array of diffusivities [m^2/s] estimated with exponential method

    Returns
    -------
    D : float
        diffusivity of CO2 in polyol [m^2/s] at the given pressure p
    """
    # averages sqrt and exponential estimates of diffusivity [m^2/s]
    D_arr = (D_sqrt_arr + D_exp_arr)/2
    # removes data points with missing measurements
    not_nan = [i for i in range(len(D_arr)) if not np.isnan(D_arr[i])]
    p_arr = p_arr[not_nan]
    D_arr = D_arr[not_nan]
    # orders saturation concentration in order of increasing pressure
    inds = np.argsort(p_arr)
    # limits the pressure to be within the minimum and maximum measured values [Pa]
    p = min(p, np.max(p_arr))
    p = max(p, np.min(p_arr))
    # interpolates diffusivity [m^2/s] to match the given pressure
    D = np.interp(p, p_arr[inds], D_arr[inds])

    return D


def calc_if_tension(p, if_interp_arrs, R, d_tolman=0):
    """
    Estimates the interfacial tension given arrays of values.

    Providing a value for the radius invokes the use of the Tolman length delta
    to correct for the effects of curvature on the interfacial tension.

    Parameters
    ----------
    p : float
        pressure at which to estimate the saturation concentration [Pa]
    p_arr : (N) numpy array of floats
        pressures [Pa]
    if_arr : (N) numpy array of floats
        interfacial tension at the pressures in p_arr [N/m].
    R : float
        radius of curvature (assumed to be same in both directions) [m].
        If R <= 0, ignored.
    d_tolman : float
        Tolman length [m]. If R <= 0, ignored.

    Returns
    -------
    if_tension : float
        interfacial tension between CO2-rich and polyol-rich phases [N/m] at the given pressure p
    """
    # interpolates interfacial tension [N/m] to match the given pressure
    if_tension = np.interp(p, *if_interp_arrs)/(1 + 2*d_tolman/R)
    return if_tension


def calc_D_of_c(c, polyol_data_file, p0=4E6):
    """
    Diffusivity as a function of concentration of CO2 in the polyol.

    Inverts calc_c_s (c_s = c_s(p)) to find the pressure at which the given
    concentration is the saturation concentration and plugs that pressure into
    calc_D (D = D(p)) to compute the diffusivity.

    Parameters
    ----------
    c : float
        concentration of CO2 in the polyol [kg CO2 / m^3 polyol-CO2]
    polyol_data_file : string
        name of file containing polyol data [.csv]

    Returns
    -------
    D : float
        diffusivity [m^2/s] of CO2 in polyol at given concentration of CO2
    """
    def f(p, c_s, polyol_data_file):
        """Root yields saturation pressure for given saturation concentration"""
        return calc_c_s(p, polyol_data_file) - c_s

    # inverts calc_c_s to get pressure [Pa] at which c is sat. conc.
    args = (c, polyol_data_file,)
    soln = scipy.optimize.root(f, p0, args=args)
    p = soln.x[0] # extracts solution from array
    # computes diffusivity at resulting pressure
    D = calc_D(p, polyol_data_file)

    return D


def calc_D_of_c_raw(c, c_s_arr, p_s_arr, p_arr, D_exp_arr, D_sqrt_arr):
    """
    Calculates D(c) as in calc_D_of_c but directly interpolates from arrays.
    """
    # pushes c in bounds if not already
    c = min(c, np.max(c_s_arr))
    c = max(c, np.min(c_s_arr))
    # interpolates saturation pressure with c as saturation concentration [Pa]
    p = np.interp(c, c_s_arr, p_s_arr)
    # estimates diffusivity [m^2/s]
    D = calc_D_raw(p, p_arr, D_exp_arr, D_sqrt_arr)

    return D


def calc_dDdc(c, dc, polyol_data_file):
    """
    Computes derivative of diffusivity of CO2 in polyol as a function of the
    concentration of CO2.

    Uses a forward difference formula to estimate dD/dc given the step size dc.

    Parameters
    ----------
    c : float
        concentration of CO2 [kg CO2 / m^3 polyol-CO2]
    dc : float
        step size in concentration of CO2 over which to evaluate the derivative
        [kg CO2 / m^3 polyol-CO2]
    polyol_data_file : string
        name of file containing polyol data [.csv]

    Returns
    -------
    dDdc : float
        derivative of diffusivity of CO2 in polyol with respect to concentration
        of CO2 [(m^2 / s) / (kg CO2 / m^3 polyol-CO2)]
    """
    # computes diffusivity at given concentration and one step size away [m^2/s]
    D1 = calc_D_of_c(c, polyol_data_file)
    D2 = calc_D_of_c(c + dc, polyol_data_file)
    # computes dD/dc using forward difference formula [m^2/s / kg/m^3]
    dDdc = (D2 - D1) / dc

    return dDdc


def calc_dDdc_fn(c, dc, D_fn):
    """
    Computes dD/dc given a functional form to estimate D(c).
    """
    # computes diffusivity at given concentration and one step size away [m^2/s]
    D1 = D_fn(c)
    D2 = D_fn(c + dc)
    # computes dD/dc using forward difference formula [m^2/s / kg/m^3]
    dDdc = (D2 - D1) / dc

    return dDdc


def calc_dDdc_raw(c, dc, c_s_arr, p_s_arr, p_arr, D_sqrt_arr, D_exp_arr):
    """
    """
    # computes diffusivity at given concentration and one step size away [m^2/s]
    D1 = calc_D_of_c_raw(c, c_s_arr, p_s_arr, p_arr, D_sqrt_arr, D_exp_arr)
    D2 = calc_D_of_c_raw(c + dc, c_s_arr, p_s_arr, p_arr, D_sqrt_arr, D_exp_arr)
    # computes dD/dc using forward difference formula [m^2/s / kg/m^3]
    dDdc = (D2 - D1) / dc

    return dDdc


def calc_D_lin(c, D_params):
    """
    Estimates diffusivity [m^2/s] as a function of the concentration c [kg/m^3]
    assuming a linear model (parameters obtained from lin_fit_D_c()).
    """
    a, b = D_params
    return a*c + b

def interp_rho_co2(eos_co2_file):
    """
    Returns an interpolation function for the density of carbon dioxide
    according to the equation of state (data taken from
    webbook.nist.gov at desired temperature.
    The density is returned in terms of kg/m^3 as a function of pressure in Pa.
    Parameters
    ----------
    eos_co2_file : string
        File name for equation of state data table [.csv]
    Returns
    -------
    rho : interpolation function
        density in kg/m^3 of co2 @ given temperature
    """
    # dataframe of appropriate equation of state (eos) data from NIST
    df_eos = pd.read_csv(eos_co2_file, header=0)
    # get list of pressures of all data points [Pa]
    p_co2 = 1000*df_eos['Pressure (kPa)'].to_numpy(dtype=float)
    # get corresponding densities of CO2 [kg/m^3]
    rho_co2 = 1000*df_eos['Density (g/ml)'].to_numpy(dtype=float)
    # remove repeated entries
    p_co2, inds_uniq = np.unique(p_co2, return_index=True)
    rho_co2 = rho_co2[inds_uniq]
    # create interpolation function [kg/m^3]
    rho_min = np.min(rho_co2)
    rho_max = np.max(rho_co2)
    f_rho_co2 = scipy.interpolate.interp1d(p_co2, rho_co2, bounds_error=False,
                                       fill_value=(rho_min, rho_max))

    return f_rho_co2


def lin_fit_D_c(c_s_arr, p_s_arr, p_arr, D_exp_arr, D_sqrt_arr):
    """
    Returns coefficients of linear fit of D(c).
    """
    # only uses squareroot fit for D since it looks more reliable
    # (see 20201125_jak_diffn.ppt, final slide)
    D_arr = D_sqrt_arr

    # removes missing entries for pressure (nans)
    not_nan, p_arr = fn.remove_nans(p_arr)
    D_arr = D_arr[not_nan]
    # removes missing entries for concentration (nans)
    not_nan, c_s_arr = fn.remove_nans(c_s_arr)
    p_s_arr = p_s_arr[not_nan]

    # only considers pressurization measurements since depressurization appeared
    # to be too fast in some cases and resulted in much higher estimates of D
    i_max = np.argmax(p_arr)
    p_fit = p_arr[:i_max+1]
    D_fit = D_arr[:i_max+1]

    # interpolates corresponding concentrations [kg/m^3]
    # TODO check that pressures are within range of p_fit
    c_fit = np.interp(p_fit, p_s_arr, c_s_arr)

    # computes linear fit
    a, b = np.polyfit(c_fit, D_fit, 1)

    return a, b


def load_c_s_arr(polyol_data_file, sort=True):
    """
    Estimates arrays of values of solubility for different pressures using
    measurements of solubility and specific volume from G-ADSA.

    If p is above the experimentally measured range, returns the maximum
    measured saturation concentration to avoid errors (this is preferable since
    we are just trying to make some rough estimates as a demonstration of this
    method right now. More precise measurements in the future will require
    a different approach).

    Units are kg/m^3 of CO2 per polyol-CO2 solution.
    """
    # loads thermophysical property data from file
    df = pd.read_csv(polyol_data_file)
    p_arr = kPa_2_Pa*df['p actual [kPa]'].to_numpy(dtype=float) # measured pressures from experiment [Pa]
    solub_arr = df['solubility [w/w]'].to_numpy(dtype=float) # measured solubility [w/w]
    spec_vol_arr = df['specific volume (fit) [mL/g]'].to_numpy(dtype=float) # fitted specific volume [mL/g]
    density_arr = gmL_2_kgm3/spec_vol_arr # density of polyol-CO2 [kg/m^3]
    # computes saturation concentration of CO2 in polyol [kg CO2 / m^3 solution]
    c_s_arr = solub_arr*density_arr

    # removes data points with missing measurements
    not_nan = [i for i in range(len(c_s_arr)) if not np.isnan(c_s_arr[i])]
    p_arr = p_arr[not_nan]
    c_s_arr = c_s_arr[not_nan]
    # concatenate 0 to pressure and saturation concentration to cover low values of p
    p_arr = np.concatenate((np.array([0]), p_arr))
    c_s_arr = np.concatenate((np.array([0]), c_s_arr))
    # orders saturation concentration in order of increasing pressure
    if sort:
        inds = np.argsort(p_arr)
        p_arr = p_arr[inds]
        c_s_arr = c_s_arr[inds]

    return p_arr, c_s_arr


def load_D_arr(polyol_data_file):
    """
    Loads arrays required for computing diffusivity D [m^2/s] as a function of
    pressure p [Pa].
    """
    # loads thermophysical property data from file
    df = pd.read_csv(polyol_data_file) # takes up 2/3 of computing time***
    p_arr = kPa_2_Pa*df['p actual [kPa]'].to_numpy(dtype=float) # measured pressures from experiment [Pa]
    D_sqrt_arr = cm2s_2_m2s*df['diffusivity (sqrt) [cm^2/s]'].to_numpy(dtype=float) # diff. measured by sqrt transient [m^2/s]
    D_exp_arr = cm2s_2_m2s*df['diffusivity (exp) [cm^2/s]'].to_numpy(dtype=float) # diff. measured by exp plateau [m^2/s]

    return p_arr, D_sqrt_arr, D_exp_arr


def load_if_tension_arr(polyol_data_file, p_min=0, p_max=4E7,
                         if_tension_model='lin'):
    """
    Estimates the interfacial tension between the CO2-rich and polyol-rich
    phases under equilibrium coexistence between CO2 and polyol at the given
    pressure by interpolating available measurements using G-ADSA. Provides
    arrays for interpolation using calc_if_tension().

    """
    # loads thermophysical property data from file
    df = pd.read_csv(polyol_data_file)
    p_arr = 1000*df['p actual [kPa]'].to_numpy(dtype=float) # measured pressures from experiment [Pa]
    if_tension_arr = 1E-3*df['if tension [mN/m]'].to_numpy(dtype=float) # measured interfacial tension [N/m]

    # removes data points with missing measurements
    not_nan = [i for i in range(len(if_tension_arr)) if not np.isnan(if_tension_arr[i])]
    p_arr = p_arr[not_nan]
    if_tension_arr = if_tension_arr[not_nan]
    # orders saturation concentration in order of increasing pressure
    inds = np.argsort(p_arr)
    p_mid = p_arr[inds]
    if_tension_mid = if_tension_arr[inds]
    # extrapolates pressure beyond range of data
    a, b = np.polyfit(p_arr, if_tension_arr, 1)
    p_small = np.linspace(p_min, p_mid[0], 10)
    if_tension_small = a*p_small + b
    p_big = np.linspace(p_mid[-1], p_max, 100)
    if if_tension_model == 'lin':
        if_tension_big = a*p_big + b
        # change negative values to 0
        if_tension_big *= np.heaviside(if_tension_big, 1)
    elif if_tension_model == 'ceil':
        if_tension_big = np.min(if_tension_mid)*np.ones([len(p_big)])
    else:
        print('load_if_tension_arr does not recognize the given if_tension_model.')

    return np.concatenate((p_small, p_mid, p_big)), \
            np.concatenate((if_tension_small, if_tension_mid, if_tension_big))


def load_w_co2_arr(polyol_data_file):
    """
    Loads arrays required for interpolating weight fraction of CO2 [w/w] as a
    function of pressure p [Pa].
    """
    # loads thermophysical property data from file
    df = pd.read_csv(polyol_data_file) # takes up 2/3 of computing time***
    p_arr = kPa_2_Pa*df['p actual [kPa]'].to_numpy(dtype=float) # measured pressures from experiment [Pa]
    w_co2_arr = df['solubility [w/w]'].to_numpy(dtype=float) # diff. measured by sqrt transient [m^2/s]
    #
    # # removes data points with missing measurements
    # not_nan = [i for i in range(len(w_co2_arr)) if not np.isnan(w_co2_arr[i])]
    # p_arr = p_arr[not_nan]
    # w_co2_arr = w_co2_arr[not_nan]
    # # concatenate 0 to pressure and saturation concentration to cover low values of p
    # p_arr = np.concatenate((np.array([0]), p_arr))
    # w_co2_arr = np.concatenate((np.array([0]), w_co2_arr))
    # # orders saturation concentration in order of increasing pressure
    # inds = np.argsort(p_arr)
    # p_arr = p_arr[inds]
    # w_co2_arr = w_co2_arr[inds]

    return p_arr, w_co2_arr
