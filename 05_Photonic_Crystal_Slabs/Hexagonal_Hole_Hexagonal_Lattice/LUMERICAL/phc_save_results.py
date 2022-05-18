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

    fcs_d1 = 335.116e12  # frequency of D1 line
    fcs_d2 = 351.726e12  # frequency of D2 line
    freq = 1.0 * fcs_d2 + 0.0 * fcs_d1
    bandwidth = 150e12  # This is giant. will it work?
    freqhighres = 400  # For more detailed plots

    # For use as input into the DipolePower() and SourcePower()
    fUse = (
        freq
        - 0.5 * bandwidth
        + bandwidth * np.linspace(0, (freqhighres - 1), freqhighres) / (freqhighres - 1)
    )

    EPurcellEnhancement = np.abs(
        fdtd.dipolepower(fUse)[:, 0] / fdtd.sourcepower(fUse)[:, 0]
    )

    moment = fdtd.getdata("DSource", "moment")
    eps_0 = 8.85e-12
    c = 3e8

    # Purcell enhancement with Green's function: E field from monitor

    E_m = fdtd.getresult("profile", "E")["E"]

    G_m = (E_m * c ** 2 * eps_0) / (
        moment * (2 * np.pi * fUse[None, None, None, :, None]) ** 2
    )

    Green_function = G_m

    P_prime_1 = np.real(fdtd.getresult("profile_prime1", "power")[:, 0])
    P_prime_2 = np.real(fdtd.getresult("profile_prime2", "power")[:, 0])
    P_prime_3 = np.real(fdtd.getresult("profile_prime3", "power")[:, 0])

    P_prime = 2 * (P_prime_1 + 2 * P_prime_2 + 2 * P_prime_3)

    Purcell_rad = np.abs(P_prime / fdtd.sourcepower(fUse)[:, 0])

    return (
        Green_function,
        EPurcellEnhancement,
        Purcell_rad,
    )


# MAIN

names = ["_phc_X", "_phc_Y", "_phc_Z"]

for i in range(3):

    name = names[i]

    results = get_results(i, name)

    np.save("results/Green_functions" + name, np.array(results[0]))
    np.savetxt("results/PurcellEnhancement" + name + ".txt", np.array(results[1]))
    np.savetxt("results/Purcell_rad" + name + ".txt", np.array(results[2]))

# %%
