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
import matplotlib

plt.style.use("science")
matplotlib.rcParams.update({"font.size": 22})


def get_results(params, i, num_iter):

    L_rad, h_rad, w_rad = params

    freqhighres = 400  # from  last convergence test

    fdtd = lumapi.FDTD(hide=True)
    fdtd.load("lumerical-running-savefile" + "_phc_convergence_box_" + str(num_iter))

    fcs_d1 = 377.107e12  # frequency of D1 line
    fcs_d2 = 384.23e12  # frequency of D2 line
    freq = 1.0 * fcs_d2 + 0.0 * fcs_d1
    bandwidth = 150e12

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

a = 353.511e-9  # Lattice constant of structure

L_rads = np.array([2, 4, 8, 11, 12, 13, 13.5]) * a
h_rads = np.array([0.2, 0.4, 0.6, 1, 1.5, 1.75, 2]) * a
w_rads = np.array([0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1]) * a

j = 0
nfreq = 400
residuals = np.zeros(len(L_rads))

for i in range(len(L_rads)):

    L_rad = L_rads[i]
    h_rad = h_rads[i]
    w_rad = w_rads[i]

    params = L_rad, h_rad, w_rad

    PF = get_results(params, j, num_iter=i)

    if i == 0:

        abs_residual = np.mean(np.abs(PF))
        rel_residual = 1.0

    else:

        abs_residual = np.mean(np.abs((PF - PF_prev)))
        rel_residual = abs_residual / np.mean(np.abs(PF))

    print("Abs. residual: ", abs_residual)
    print("Rel. residual: ", rel_residual)
    residuals[i] = rel_residual
    abs_residual_prev = abs_residual
    PF_prev = PF
    n_freq_prev = nfreq
# %%

matplotlib.rcParams.update({"font.size": 28})

iterations = np.linspace(0, len(residuals), len(residuals))

fig, ax = plt.subplots(figsize=(16, 8))


ax.plot(iterations, residuals, color="red", linewidth=3, linestyle="dashed")
ax.scatter(iterations, residuals, color="red", marker="o", linewidth=10)
ax.set_xlabel("Iteration num.")
ax.set_ylabel("Relative residual")
ax.set_yscale("log")
ax.set_title("Box convergence test", pad=+10)

# %%

fig.savefig("relative_residual_phc.pdf")

# %%
