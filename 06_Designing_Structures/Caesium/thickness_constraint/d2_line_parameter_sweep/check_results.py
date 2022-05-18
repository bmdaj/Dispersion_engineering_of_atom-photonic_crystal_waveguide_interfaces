#%%
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

plt.style.use("science")
mpl.rcParams.update({"font.size": 46})

# %%

names = ["X"]
aPhC_list = np.linspace(300e-9, 450e-9, 15)
multiplier_list = np.linspace(0.1, 0.35, 10)


freq = 351.726e12
bandwidth = 150e12
freqhighres = 400

fUse = (
    freq
    - 0.5 * bandwidth
    + bandwidth * np.linspace(0, (freqhighres - 1), freqhighres) / (freqhighres - 1)
) * 10 ** -12

f_cs_d2_line = 351.73  # THz
cost = np.zeros((15, 10))
coop_mat = np.zeros((15, 10))
pf_mat = np.zeros((15, 10))


for j in range(len(aPhC_list)):
    for k in range(len(multiplier_list)):
        name = "X" + "_" + str(j) + "_" + str(k)
        print(name)
        Purcell_X = np.loadtxt("results/PurcellEnhancement" + str(name) + ".txt")
        Purcell_rad_X = np.loadtxt("results/Purcell_rad" + str(name) + ".txt")
        Purcell_1D_X = Purcell_X - Purcell_rad_X
        Gamma_1D_vs_prime_X = Purcell_1D_X / Purcell_rad_X
        freq_max = fUse[list(Gamma_1D_vs_prime_X).index(np.max(Gamma_1D_vs_prime_X))]
        print(freq_max)
        coop_mat[j, k] = np.max(Gamma_1D_vs_prime_X)
        cost[j, k] = np.abs(freq_max - f_cs_d2_line)
        pf_mat[j, k] = Purcell_X[
            list(Gamma_1D_vs_prime_X).index(np.max(Gamma_1D_vs_prime_X))
        ]
# %%
k = 0
for j in range(len(multiplier_list)):
    print(coop_mat[k, j])
# %%
fig, ax = plt.subplots(figsize=(14, 8))


name = "X" + "_" + str(8) + "_" + str(4)
print(name)
Purcell_X = np.loadtxt("results/PurcellEnhancement" + str(name) + ".txt")
Purcell_rad_X = np.loadtxt("results/Purcell_rad" + str(name) + ".txt")
Purcell_1D_X = Purcell_X - Purcell_rad_X
Gamma_1D_vs_prime_X = Purcell_1D_X / Purcell_rad_X
print(np.max(Gamma_1D_vs_prime_X))
freq_max = fUse[list(Gamma_1D_vs_prime_X).index(np.max(Gamma_1D_vs_prime_X))]
print(freq_max)
ax.plot(fUse, Gamma_1D_vs_prime_X)
print(Purcell_X[list(Gamma_1D_vs_prime_X).index(np.max(Gamma_1D_vs_prime_X))])
ax.set_yscale("log")
# %%
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


fig, ax = plt.subplots(1, 3, figsize=(32, 8))


aPhC_list_new = aPhC_list * 1e9

extent = [aPhC_list_new[0], aPhC_list_new[-2], multiplier_list[-1], multiplier_list[0]]
im_pf = ax[0].imshow(pf_mat.T, extent=extent, aspect="auto")

im0 = ax[1].imshow(
    coop_mat.T, extent=extent, aspect="auto"
)  # , norm=LogNorm(vmin=0.01)

extent = [aPhC_list_new[0], aPhC_list_new[-2], multiplier_list[-1], multiplier_list[0]]
im_pf = ax[0].imshow(pf_mat.T, extent=extent, aspect="auto")
ax[1].set_xlabel("$a \,(nm)$")
ax[0].set_ylabel("Hole factor")
ax[0].set_title("Purcell Factor")
ax[1].set_title("Maximum cooperativity")
im1 = ax[2].imshow(cost.T, aspect="auto", extent=extent)
ax[2].set_xlabel("$a \,(nm)$")
# ax[1].set_ylabel("Hole factor")
ax[2].set_title("Detuning from D2 line in THz")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im_pf, cax=cax, orientation="vertical")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im0, cax=cax, orientation="vertical")
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax, orientation="vertical")
fig.savefig("param_sweep_results.pdf")
# %%

print(aPhC_list[8])
print(multiplier_list[4])

# %%

fig, ax = plt.subplots(figsize=(12, 10))
extent = [aPhC_list_new[0], aPhC_list_new[-2], multiplier_list[-1], multiplier_list[0]]
im_pf = ax.imshow(pf_mat.T, extent=extent, aspect="auto")
extent = [aPhC_list_new[0], aPhC_list_new[-2], multiplier_list[-1], multiplier_list[0]]
ax.set_ylabel("Hole factor")
ax.set_xlabel("$a \,(nm)$")
divider = make_axes_locatable(ax)
# ax.set_title("Purcell Factor")
ax.tick_params(pad=15)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im_pf, cax=cax, orientation="vertical")
fig.savefig("param_sweep_results_pf.pdf")

# %%
fig, ax = plt.subplots(figsize=(12, 10))
extent = [aPhC_list_new[0], aPhC_list_new[-2], multiplier_list[-1], multiplier_list[0]]
im_pf = ax.imshow(coop_mat.T, extent=extent, aspect="auto")
extent = [aPhC_list_new[0], aPhC_list_new[-2], multiplier_list[-1], multiplier_list[0]]
ax.set_ylabel("Hole factor")
ax.set_xlabel("$a \,(nm)$")
# ax.set_title("Maximum cooperativity")
divider = make_axes_locatable(ax)
ax.tick_params(pad=15)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im_pf, cax=cax, orientation="vertical")
fig.savefig("param_sweep_results_coop.pdf")

# %%
fig, ax = plt.subplots(figsize=(12, 10))
extent = [aPhC_list_new[0], aPhC_list_new[-2], multiplier_list[-1], multiplier_list[0]]
im_pf = ax.imshow(cost.T, extent=extent, aspect="auto")
extent = [aPhC_list_new[0], aPhC_list_new[-2], multiplier_list[-1], multiplier_list[0]]
ax.set_ylabel("Hole factor")
ax.set_xlabel("$a \,(nm)$")
# ax.set_title("Detuning from D2 line in THz")
divider = make_axes_locatable(ax)
ax.tick_params(pad=15)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im_pf, cax=cax, orientation="vertical")
fig.savefig("param_sweep_results_det.pdf")
# %%
