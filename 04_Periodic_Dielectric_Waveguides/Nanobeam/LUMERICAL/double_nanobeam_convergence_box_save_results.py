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

    l_b, w_b, h_b = params

    freqhighres = 400  # from  last convergence test

    fdtd = lumapi.FDTD(hide=True)
    fdtd.load("lumerical-running-savefile" + "_convergence_" + str(num_iter))

    freq = 350e12  # Choice for center frequency of pulse
    bandwidth = 250e12

    fUse = (
        freq
        - 0.5 * bandwidth
        + bandwidth * np.linspace(0, (freqhighres - 1), freqhighres) / (freqhighres - 1)
    )

    P_prime_1 = np.real(fdtd.getresult("profile_prime1", "power")[:, 0])
    P_prime_3 = np.real(fdtd.getresult("profile_prime3", "power")[:, 0])

    P_prime = 2 * (P_prime_1 + P_prime_3)

    Purcell_rad = np.abs(P_prime / fdtd.sourcepower(fUse)[:, 0])

    return Purcell_rad


# %%
# -----------------------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------------------

a = 335e-9  # Lattice constant of structure

l_bs = np.array([5, 10, 20, 30, 40]) * a
w_bs = np.array([0.5, 1, 2, 4, 8]) * a
h_bs = np.array([0.25, 0.5, 1, 2, 4]) * a

nfreq = 400  # from last convergence test

j = 0  # convergence test on X dipole orientation

f = open("convergence_box.log", "w")  # logfile to track progress of simulation

f.write("--------------------------------------------------\n")
f.write("CONVERGENCE LOGFILE\n")
f.write("--------------------------------------------------\n")
f.write(
    "We track the progress of the parameters and the convergence of the simulation for the FOM: \n"
)
f.write("PURCELL FACTOR \n")
f.write("--------------------------------------------------\n")
f.write("COLUMNS: \n")
f.write("PF-rad, l-b, w-b, h-b, Absolute Residual, Relative residual\n")

for i in range(len(l_bs)):

    l_b = l_bs[i]
    w_b = w_bs[i]
    h_b = h_bs[i]

    params = l_b, w_b, h_b

    PF = get_results(params, j, i)

    np.savetxt(
        "convergence_results/PF-rad_l-b_"
        + str(l_b)
        + "_w-b_"
        + str(w_b)
        + "_h-b_"
        + str(h_b)
        + ".dat",
        PF,
    )

    if i == 0:

        abs_residual = np.sum(np.abs(PF)) / nfreq
        rel_residual = 1.0

    else:

        abs_residual = np.abs(
            np.sum(np.abs(PF)) / nfreq - np.sum(np.abs(PF_prev)) / n_freq_prev
        )
        rel_residual = abs_residual / (np.sum(np.abs(PF)) / nfreq)
        print(rel_residual)

    abs_residual_prev = abs_residual
    PF_prev = PF
    n_freq_prev = nfreq

    f.write(
        str(l_b)
        + ","
        + str(w_b)
        + ","
        + str(h_b)
        + ","
        + str("{:010.4f}".format(abs_residual))
        + ","
        + str("{:01.5f}".format(rel_residual) + "\n")
    )
    # %%
