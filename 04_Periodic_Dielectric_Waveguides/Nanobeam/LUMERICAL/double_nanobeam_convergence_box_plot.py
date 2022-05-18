# %%

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


def get_results(params, i):

    mesh_acc, _, nfreq = params

    freqhighres = nfreq

    fdtd = lumapi.FDTD(hide=True)
    fdtd.load(
        "lumerical-running-savefile" + "_convergence_mesh_acc_" + str(int(mesh_acc))
    )

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


mesh_accs = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
time_factors = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
nfreqs = np.array([100, 200, 300, 400, 500, 600, 700])
residuals_0 = np.zeros(len(mesh_accs))
j = 0

for i in range(len(mesh_accs)):

    mesh_acc = mesh_accs[i]
    time_factor = time_factors[i]
    nfreq = nfreqs[i]

    params = mesh_acc, time_factor, nfreq

    PF = get_results(params, j)

    if i == 0:

        abs_residual = np.sum(np.abs(PF)) / nfreq
        rel_residual = 1.0

    else:

        abs_residual = np.abs(
            np.sum(np.abs(PF)) / nfreq - np.sum(np.abs(PF_prev)) / n_freq_prev
        )
        rel_residual = abs_residual / (np.sum(np.abs(PF)) / nfreq)

    print("Abs. residual: ", abs_residual)
    print("Rel. residual: ", rel_residual)
    residuals_0[i] = rel_residual

    abs_residual_prev = abs_residual
    PF_prev = PF
    n_freq_prev = nfreq


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

a = 335e-9  # Lattice constant of structure

l_bs = np.array([5, 10, 20, 30, 40]) * a
w_bs = np.array([0.5, 1, 2, 4, 8]) * a
h_bs = np.array([0.25, 0.5, 1, 2, 4]) * a

nfreq = 400  # from last convergence test

j = 0  # convergence test on X dipole orientation
residuals = np.zeros(len(l_bs))

for i in range(len(l_bs)):

    l_b = l_bs[i]
    w_b = w_bs[i]
    h_b = h_bs[i]

    params = l_b, w_b, h_b

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

matplotlib.rcParams.update({"font.size": 36})

iterations = np.linspace(0, len(residuals), len(residuals))

fig, ax = plt.subplots(1, 2, figsize=(24, 8))

ax[0].scatter(
    np.linspace(0, len(residuals_0) - 1, len(residuals_0)),
    residuals_0,
    color="blue",
    marker="o",
    linewidth=10,
)
ax[0].scatter(
    [3],
    residuals_0[3],
    color="red",
    marker="o",
    linewidth=20,
)
ax[0].plot(
    np.linspace(0, len(residuals_0) - 1, len(residuals_0)),
    residuals_0,
    color="blue",
    linewidth=3,
    linestyle="dashed",
)
ax[1].plot(iterations, residuals, color="red", linewidth=3, linestyle="dashed")
ax[1].scatter(iterations, residuals, color="red", marker="o", linewidth=10)
ax[1].set_xlabel("Iteration number")
ax[0].set_xlabel("Iteration number")
ax[0].set_ylabel("$\delta$PF")
ax[1].set_ylabel("$\delta\\Gamma^\prime/\\Gamma_0$")
ax[0].set_yscale("log")
ax[1].set_yscale("log")
ax[0].set_title("PF convergence test", pad=+14)
ax[1].set_title("Box convergence test", pad=+14)
ax[0].grid(True, which="both", alpha=1)
ax[1].grid(True, which="both", alpha=1)
# %%

fig.savefig("relative_residual_nanobeam.pdf")

# %%
