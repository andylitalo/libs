"""
@brief diffusion.py plots diffusion data.

@author Andy Ylitalo
@date October 22, 2020
"""

import matplotlib.pyplot as plt
from matplotlib import cm # colormap
import numpy as np

# imports conversions
from plot.conversions import *



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


def get_color(cmap_name, val, val_list):
    """Returns list of colors using given colormap."""
    cmap = plt.get_cmap(cmap_name)
    num = val - np.min(val_list)
    den = np.max(val_list) - np.min(val_list)

    return cmap(num/den)


def sheath(t, r_arr, c, R_i, R_o, v, c_s, n_plot, ax=None, t_fs=18,
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
