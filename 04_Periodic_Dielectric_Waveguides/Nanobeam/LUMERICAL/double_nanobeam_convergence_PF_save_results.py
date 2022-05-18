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
def get_results(params, i):

    mesh_acc, _, nfreq = params

    freqhighres = nfreq

    fdtd = lumapi.FDTD(hide=True)
    fdtd.load(
        "lumerical-running-savefile" + "_convergence_mesh_acc_" + str(int(mesh_acc))
    )

    print(mesh_acc)

    freq = 350e12  # Choice for center frequency of pulse
    bandwidth = 250e12

    fUse = (
        freq
        - 0.5 * bandwidth
        + bandwidth * np.linspace(0, (freqhighres - 1), freqhighres) / (freqhighres - 1)
    )

    EPurcellEnhancement = np.abs(
        fdtd.dipolepower(fUse)[:, 0] / fdtd.sourcepower(fUse)[:, 0]
    )

    return EPurcellEnhancement


# %%
# -----------------------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------------------

mesh_accs = np.array([2.0, 3.0, 4.0])  # , 5.0, 6.0])
time_factors = np.array([50.0, 100.0, 200.0])  # , 800.0, 1600])
nfreqs = np.array([200, 400, 800])  # , 800, 1600])

j = 0  # convergence test on X dipole orientation

f = open("convergence_PF.log", "w")  # logfile to track progress of simulation

f.write("--------------------------------------------------\n")
f.write("CONVERGENCE LOGFILE\n")
f.write("--------------------------------------------------\n")
f.write(
    "We track the progress of the parameters and the convergence of the simulation for the FOM: \n"
)
f.write("PURCELL FACTOR \n")
f.write("--------------------------------------------------\n")
f.write("COLUMNS: \n")
f.write("mesh accuracy, time factor, nfreq, Absolute Residual, Relative residual\n")

for i in range(len(mesh_accs)):

    mesh_acc = mesh_accs[i]
    time_factor = time_factors[i]
    nfreq = nfreqs[i]

    params = mesh_acc, time_factor, nfreq

    PF = get_results(params, j)

    np.savetxt(
        "convergence_results/PF_mesh-acc_"
        + str(mesh_acc)
        + "_time-factor_"
        + str(time_factor)
        + "_nfreq_"
        + str(nfreq)
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
        str(mesh_acc)
        + ","
        + str(time_factor)
        + ","
        + str(nfreq)
        + ","
        + str("{:010.4f}".format(abs_residual))
        + ","
        + str("{:01.5f}".format(rel_residual) + "\n")
    )
# %%
