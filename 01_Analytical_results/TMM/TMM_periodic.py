# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.style.use("science")
matplotlib.rcParams.update({'font.size': 22})


import sys

sys.path.append(".")

from TMM_class import TMM

# %%

a = 380e-9  # lattice constant of photonic crystal structure
# If in Y direction a=650e-9
na = 1
nb = 2  # SiN
La = 0.3 * a  # Hole lenght
Lb = 0.7 * a  # Dielectric lentgh
N_cells = 60  # 20 unit cells in one axis
N_taper = 0  # number of tapering unit cells added from left and right
freqs = np.linspace(360, 373, 1000) * 1e12  # frequencies we want to scan

TMM_phc = TMM(
    freqs=freqs,
    na=na,
    La=La,
    N_cells=N_cells,
    N_taper=N_taper,
    a=a,
    nb=nb,
    Lb=Lb,
    type="Periodic",
)

# %%

beta = TMM_phc.dispersion_relation()

# %%
# TO BE CHECKED: UNITS ARE NOT NORMALIZED

fig, ax = plt.subplots(1, 2, figsize=(16, 4))

ax[0].plot(freqs * a / 3e8, np.real(beta) * a)
ax[1].plot(freqs * a / 3e8, np.abs(np.imag(beta)) * a)

ax[0].set_xlabel("$\\omega$ a / 2 $\\pi$ c")
ax[0].set_ylabel("k a / 2 $\\pi$")
ax[1].set_xlabel("$\\omega$ a / 2 $\\pi$ c")
ax[1].set_ylabel("k a / 2 $\\pi$")

# %%

t = TMM_phc.calculate_transmission()
r = TMM_phc.calculate_reflection()

# %%

fig, ax = plt.subplots(1, 2, figsize=(10, 6))

ax[0].plot(freqs, t)
ax[1].plot(freqs, r)


ax[0].set_xlabel("freq")
ax[0].set_ylabel("t")
ax[1].set_xlabel("freq")
ax[1].set_ylabel("r")

# %%

vg = TMM_phc.group_velocity()
avg_vg = TMM_phc.mean_group_velocity()
res_freq, est_vg = TMM_phc.group_velocity_estimation()

# %%

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(freqs[1:], vg, color="red", label="Analytical")
ax.plot(freqs[1:], avg_vg / 3e8, color="blue", label="Mean")
ax.scatter(res_freq, est_vg / 3e8, color="green", label="Estimated")


ax.set_xlabel("freq")
ax.set_ylabel("vg (c units)")
ax.legend()
ax.set_ylim(np.min(avg_vg) / 3e8 - 0.1, np.max(avg_vg) / 3e8 + 0.1)

# %%

ng = 1 / vg
avg_ng = 3e8 / avg_vg
est_ng = 3e8 / est_vg

# %%

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(freqs[1:], ng, color="red", label="Analytical")
ax.plot(freqs[1:], avg_ng, color="blue", label="Mean")
ax.scatter(res_freq, est_ng, color="green", label="Estimated")


ax.set_xlabel("freq")
ax.set_ylabel("ng")
ax.legend()
# ax[0].set_ylim(1,200)
# ax[1].set_ylim(1, 5)

# %%

N_cells_array = np.arange(3, 100)
I = np.zeros((len(freqs), len(N_cells_array)))

for i, N_cells in enumerate(N_cells_array, start=0):
    TMM_phc = TMM(
        freqs=freqs,
        na=na,
        La=La,
        N_cells=int(N_cells),
        a=a,
        nb=nb,
        Lb=Lb,
        type="Periodic",
    )
    I[:, i] = TMM_phc.calculate_intensity_profile()

# %%

fig, ax = plt.subplots(figsize=(10, 6))

ax.contourf(
    I,
    extent=[np.min(N_cells_array), np.max(N_cells_array), np.min(freqs), np.max(freqs)],
)
ax.set_xlabel("N cells")
ax.set_ylabel("freq")
# %%
