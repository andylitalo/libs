"""
@brief diffusion.py plots diffusion data.

@author Andy Ylitalo
@date October 22, 2020
"""

import matplotlib.pyplot as plt
from matplotlib import cm # colormap
import numpy as np

import plot.genl as pltg
import plot.bubble as pltb

# imports conversions
from plot.conversions import *


def bubble_sheath(R, m, p, p_bub, rho_co2, if_tension, t_num, t_nuc,
                        t_flow, r_arr, c, R_max, v, c_bulk, n_plot, R_i=0):
    """Plots bubble properties over time and sheath concentration profile."""
    props_list_num = (R, m, p, p_bub, rho_co2, if_tension)
    ax1 = pltb.all_props(t_num, t_nuc, props_list_num, x_log=True, title='Numerical')
    ax2 = sheath(t_flow, r_arr, c, R_i, R_max, v, c_bulk, n_plot)
    # plots legend outside box
    pltg.legend(ax2)

    return ax1, ax2


def compare_sheath_Qi(d, r_arr, c_list, Q_i_list, R_i_list, R_o, v_list, c_s,
                    ax=None, t_fs=18, a_fs=16, tk_fs=14, l_fs=12, cmap_name='brg'):
    """Plots diffusion from inner to outer stream in sheath flow vs. Qi."""
    if ax is None:
        # creates plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # defines plot limits
    x_lim = [0, R_o*m_2_um]
    y_lim = [0, 1.1*c_s]

    for c, Q_i, R_i, v in zip(c_list, Q_i_list, R_i_list, v_list):
        color = get_color(cmap_name, Q_i, Q_i_list)
        t = d/v
        ax.plot(r_arr*m_2_um, c, color=color,
                    label='{0:d}'.format(int(Q_i)) + r' $\mu$L/min, '
                                        '{0:d} ms'.format(int(t*s_2_ms)))
        ax.plot([R_i*m_2_um, R_i*m_2_um], y_lim, '--', color=color,
                label=r'$R_i$' + ' = {0:d} '.format(int(R_i*m_2_um)) + r'$\mu$m')

    # formats plot
    ax.set_title('CO2 Concentration vs. Radius in \n' + \
                '{0:d} um ID Capillary at {1:d} mm\n'.format(int(2*R_o*m_2_um), int(d*m_2_mm)) + \
                'vs. Inner Stream Flow Rate', fontsize=t_fs)
    ax.set_xlabel(r'$r$ [$\mu$m]', fontsize=a_fs)
    ax.set_ylabel(r'$c_{CO2}$ [$kg$/$m^3$]', fontsize=a_fs)
    ax.tick_params(axis='both', labelsize=tk_fs)
    plt.legend(fontsize=l_fs)
    # sets limits based on values calculated above
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    return ax


def diff(t_eps, diff_list, label_list, t_halved_list=None,
                    cmap_name='winter'):
    """Plots discrepancy of a property from Epstein-Plesset model."""
    # plots difference
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # shifts time by nucleation time and converts to [ms]
    t = (np.asarray(t_eps) - t_eps[0]) * s_2_ms

    # gets colormap
    n = len(diff_list)
    color_list = pltg.get_colors(cmap_name, n)

    for i, label in enumerate(label_list):
        ax.plot(t, diff_list[i], color=color_list[i], label=label)
        if t_halved_list is not None:
            y_lim = ax.get_ylim()
            for t_halved in t_halved_list[i]:
                t_halved *= s_2_ms
                ax.plot([t_halved, t_halved], y_lim, '--', color=color_list[i])


    ax.set_xlabel(r'$t$ [ms]', fontsize=16)
    # default vertical axis in discrepancy in dc/dr
    ax.set_ylabel(r'$\Delta \left(\frac{dc}{dr}\right)$ /' + \
                    r' $\left(\frac{dc}{dr}\right)_{EP}$', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.legend()

    return ax


def get_color(cmap_name, val, val_list):
    """Returns list of colors using given colormap."""
    cmap = plt.get_cmap(cmap_name)
    num = val - np.min(val_list)
    den = np.max(val_list) - np.min(val_list)

    return cmap(num/den)


def R_diff(t1, R1, t2, R2, ax_fs=16):
    """Plots fractional discrepancy in the predicted radius from R1."""
    # initializes plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    R1 = np.asarray(R1)
    # interpolates R2 to be on same time axis as R1
    R2 = np.interp(t1, t2, np.asarray(R2))
    # computes fractional difference
    R_frac_diff = np.abs(R1 - R2) / R1

    # plot and formatting
    ax.plot(np.asarray(t1) - t1[0], R_frac_diff)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t$ [s]', fontsize=ax_fs)
    ax.set_ylabel(r'$\Delta R/ R_{eps}$', fontsize=ax_fs)
    ax.set_ylim([1E-3, 1])

    return ax



def sheath(t, r_arr_, c, R_i, R_o, v, c_s, n_plot, ax=None, t_fs=18,
                    ax_fs=16, tk_fs=14, l_fs=12):
    """Plots diffusion from inner stream to outer stream in sheath flow."""
    if ax is None:
        # initializes plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # defines plot limits
    x_lim = [0, R_o*m_2_um]
    y_lim = [0, 1.1*c_s]

    # plot desired concentration profiles
    n_steps = len(t)
    skip = int(n_steps / n_plot)

    # plots CO2 concentration profile at evenly spaced time points
    for i in range(0, n_steps, skip):
        # in case grid changes and a list of different grids is provided,
        # extracts the corresponding grid
        if isinstance(r_arr_, list):
            r_arr = r_arr_[i]
        # otherwise, uses single provided grid
        else:
            r_arr = r_arr_
        ax.plot(r_arr*m_2_um, c[i], label='t = {0:.6f} ms, d = {1:d} mm' \
                .format(t[i]*s_2_ms, int(v*t[i]*m_2_mm)))
    # plots vertical dashed line marking inner radius
    ax.plot([R_i*m_2_um, R_i*m_2_um], y_lim, 'k--', label=r'$R_i$' + \
                ' = {0:d} '.format(int(R_i*m_2_um)) + r'$\mu$m')

    # formats plot
    ax.set_title('Diffusion of CO2 from Inner to Outer Stream\n' + \
                'VORANOL 360, {0:d} um ID Capillary'.format(int(2*R_o*m_2_um)),
                 fontsize=t_fs)
    ax.set_xlabel(r'$r$ [$\mu$m]', fontsize=ax_fs)
    ax.set_ylabel(r'$c_{CO2}$ [$kg$/$m^3$]', fontsize=ax_fs)
    ax.tick_params(axis='both', labelsize=tk_fs)
    plt.legend(fontsize=l_fs)
    # sets limits based on values calculated above
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    return ax
