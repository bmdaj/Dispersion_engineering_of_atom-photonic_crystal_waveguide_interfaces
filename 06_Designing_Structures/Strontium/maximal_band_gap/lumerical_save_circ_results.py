# %%
# -----------------------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------------------
import sys, os
import subprocess

sys.path.append("/opt/lumerical/v212/api/python/")  # Default windows lumapi path
sys.path.append("/opt/lumerical/v212/api/")
sys.path.append("/opt/lumerical/v212/bin/")
sys.path.append("/home/bmda/anaconda3/envs/pmp/lib/python3.9/site-packages/")
sys.path.append(os.path.dirname(__file__))  # Current directory

import lumapi
import numpy as np
import matplotlib.pyplot as plt


def get_results(i, name):

    fdtd = lumapi.FDTD(hide=True)
    fdtd.load("lumerical-running-savefile" + str(name))

    fcs = 166.551e12
    freq = 1.0 * fcs
    bandwidth = 100e12  # This is giant. will it work?
    freqres = 5  # Freq to monitor - low res
    freqhighres = 250  # For more detailed plots

    # For use as input into the DipolePower() and SourcePower()
    fUse = (
        freq
        - 0.5 * bandwidth
        + bandwidth * np.linspace(0, (freqhighres - 1), freqhighres) / (freqhighres - 1)
    )

    # Purcell enhancement with Green's function: E field from monitor

    E_m = fdtd.getresult("profile", "E")["E"]
    index = fdtd.getresult("index", "index_z")

    return (E_m, index)


# MAIN

names = ["_phc_circ"]

for i in range(1):

    name = names[i]

    results = get_results(i, name)

    np.save("results/Green_functions" + name, np.array(results[0]))
    np.save("results/index" + name, np.array(results[1]))

# %%
