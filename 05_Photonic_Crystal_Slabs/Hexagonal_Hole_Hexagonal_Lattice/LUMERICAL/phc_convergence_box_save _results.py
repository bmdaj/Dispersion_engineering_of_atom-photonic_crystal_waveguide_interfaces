# %%
# -----------------------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------------------
import sys, os
import subprocess
import time

sys.path.append("/opt/lumerical/v212/api/python/")  # Default windows lumapi path
sys.path.append("/opt/lumerical/v212/api/")
sys.path.append("/opt/lumerical/v212/bin/")
sys.path.append("/home/bmda/anaconda3/envs/pmp/lib/python3.9/site-packages/")
sys.path.append(os.path.dirname(__file__))  # Current directory

import lumapi
import numpy as np
import matplotlib.pyplot as plt

# %%


def get_results(params, i, num_iter):

    L_rad, h_rad, w_rad = params

    freqhighres = 800  # from  last convergence test

    fdtd = lumapi.FDTD(hide=True)
    fdtd.load("lumerical-running-savefile" + "_phc_convergence_box_" + str(num_iter))

    freq = 350e12  # Choice for center frequency of pulse
    bandwidth = 250e12

    fUse = (
        freq
        - 0.5 * bandwidth
        + bandwidth * np.linspace(0, (freqhighres - 1), freqhighres) / (freqhighres - 1)
    )

    P_prime_1 = np.real(fdtd.getresult("profile_prime1", "power")[:, 0])
    P_prime_2 = np.real(fdtd.getresult("profile_prime2", "power")[:, 0])
    P_prime_3 = np.real(fdtd.getresult("profile_prime3", "power")[:, 0])

    P_prime = 2 * (P_prime_1 + 2 * P_prime_2 + 2 * P_prime_3)

    Purcell_rad = np.abs(P_prime / fdtd.sourcepower(fUse)[:, 0])

    return Purcell_rad


# %%
# -----------------------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------------------

a = 410e-9  # Lattice constant of structure

L_rads = np.array([2, 4, 8, 11, 12, 13, 13.5]) * a
h_rads = np.array([0.1, 0.2, 0.3, 0.5, 0.75, 0.8, 0.85]) * 1e-6
w_rads = np.array([0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]) * 1e-6

nfreq = 800  # from last convergence test

j = 0  # convergence test on Z dipole orientation

f = open("phc_convergence_box.log", "w")  # logfile to track progress of simulation

f.write("--------------------------------------------------\n")
f.write("CONVERGENCE LOGFILE\n")
f.write("--------------------------------------------------\n")
f.write(
    "We track the progress of the parameters and the convergence of the simulation for the FOM: \n"
)
f.write("PURCELL FACTOR \n")
f.write("--------------------------------------------------\n")
f.write("COLUMNS: \n")
f.write("PF-rad, L, h, w, Absolute Residual, Relative residual\n")


for i in range(len(L_rads)):

    L_rad = L_rads[i]
    h_rad = h_rads[i]
    w_rad = w_rads[i]

    params = L_rad, h_rad, w_rad

    PF = get_results(params, j, num_iter=i)

    np.savetxt(
        "convergence_results/PF-rad_l-b_"
        + str(L_rad)
        + "_w-b_"
        + str(h_rad)
        + "_h-b_"
        + str(w_rad)
        + ".dat",
        PF,
    )

    if i == 0:

        abs_residual = np.mean(np.abs(PF))
        rel_residual = 1.0

    else:

        abs_residual = np.mean(np.abs((PF - PF_prev)))
        rel_residual = abs_residual / np.mean(np.abs(PF))
        print((np.sum(np.abs(PF))))
        print(abs_residual)
        print(rel_residual)

    abs_residual_prev = abs_residual
    PF_prev = PF
    n_freq_prev = nfreq

    f.write(
        str(L_rad)
        + ","
        + str(h_rad)
        + ","
        + str(w_rad)
        + ","
        + str("{:010.4f}".format(abs_residual))
        + ","
        + str("{:01.5f}".format(rel_residual) + "\n")
    )
    # %%
