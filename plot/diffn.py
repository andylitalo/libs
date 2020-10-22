"""
@brief diffusion.py plots diffusion data.

@author Andy Ylitalo
@date October 22, 2020
"""

import matplotlib.pyplot as plt
import numpy as np

# imports conversions
from plot.conversions import *

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
        ax.plot(r_arr*m_2_um, c[i], label='t = {0:d} ms, d = {1:d} mm' \
                .format(int(t[i]*s_2_ms), int(v*t[i]*m_2_mm)))
    # plots vertical dashed line marking inner radius
    ax.plot([R_i*m_2_um, R_i*m_2_um], y_lim, 'k--', label=r'$R_i$')

    # formats plot
    ax.set_title('Diffusion of CO2 from Inner to Outer Stream\n' + \
                'VORANOL 360, {0:d} um ID Capillary'.format(int(2*R_o*m_2_um)),
                 fontsize=t_fs)
    ax.set_xlabel(r'$r$ [$\mu$m]', fontsize=ax_fs)
    ax.set_ylabel(r'$c_{CO2}$ [$kg^3$/$m^3$]', fontsize=ax_fs)
    ax.tick_params(axis='both', labelsize=tk_fs)
    plt.legend(fontsize=l_fs)
    # sets limits based on values calculated above
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    return ax
