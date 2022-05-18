#%%
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib

plt.style.use("science")
matplotlib.rcParams.update({"font.size": 42})


def reshape_dimension(array, dimension):

    from scipy.ndimage.interpolation import map_coordinates

    A = array

    new_dims = []
    for original_length, new_length in zip(A.shape, [dimension]):
        new_dims.append(np.linspace(0, original_length - 1, new_length))

    coords = np.meshgrid(*new_dims, indexing="ij")
    B = map_coordinates(A, coords)

    return B


#%%

Purcell_X = np.loadtxt("results/PurcellEnhancement_nb_X.txt")
Purcell_Y = np.loadtxt("results/PurcellEnhancement_nb_Y.txt")
Purcell_Z = np.loadtxt("results/PurcellEnhancement_nb_Z.txt")

freq = 350e12  # Choice for center frequency of pulse
bandwidth = 250e12
freqhighres = 10000
# For more detailed plots

fUse = (
    freq
    - 0.5 * bandwidth
    + bandwidth * np.linspace(0, (freqhighres - 1), freqhighres) / (freqhighres - 1)
) * 10 ** -12

fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(fUse, Purcell_X, color="red", label="X orientation", lw=2)
ax.plot(fUse, Purcell_Y, color="blue", label="Y orientation", lw=2)
ax.plot(fUse, Purcell_Z, color="green", label="Z orientation", lw=2)

# ax.set_yscale("log")

ax.set_ylabel("PF $\equiv \Gamma_{\\text{tot}}/\Gamma_0$")
ax.set_ylim(0, 25)
ax.set_xlabel("$\\nu$ (THz)")

ax.legend(frameon=True)
fig.savefig("results/Purcell_nb.pdf")
# %%

# mpl.rcParams.update({"font.size": 24})
freq = 350e12  # Choice for center frequency of pulse
bandwidth = 20e12
freqhighres = 10000  # For more detailed plots

fUse = (
    freq
    - 0.5 * bandwidth
    + bandwidth * np.linspace(0, (freqhighres - 1), freqhighres) / (freqhighres - 1)
) * 10 ** -12

fig, ax = plt.subplots(figsize=(10, 8))

Purcell_Y_zoom = np.loadtxt("results/PurcellEnhancement_zoom_nb_Y_zoom.txt")

ax.plot(fUse, Purcell_Y_zoom, color="blue", label="Y orientation", lw=2)

ax.set_ylabel("PF $\equiv \Gamma_{\\text{tot}}/\Gamma_0$")
ax.set_xlabel("$\\nu$  (THz)")
# ax.set_yscale("log")
# ax.set_ylim(0, 30)
# ax.legend(loc=2, frameon=True)

fig.savefig("results/Purcell_nb_Y_zoom.pdf")
# %%

freq = 350e12  # Choice for center frequency of pulse
bandwidth = 250e12
freqhighres = 10000  # For more detailed plots

fUse = (
    freq
    - 0.5 * bandwidth
    + bandwidth * np.linspace(0, (freqhighres - 1), freqhighres) / (freqhighres - 1)
) * 10 ** -12


Purcell_rad_X = np.loadtxt("results/Purcell_rad_nb_X.txt")
Purcell_rad_Y = np.loadtxt("results/Purcell_rad_nb_Y.txt")
Purcell_rad_Z = np.loadtxt("results/Purcell_rad_nb_Z.txt")

Purcell_rad_X = reshape_dimension(Purcell_rad_X, freqhighres)
Purcell_rad_Y = reshape_dimension(Purcell_rad_Y, freqhighres)
Purcell_rad_Z = reshape_dimension(Purcell_rad_Z, freqhighres)

fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(fUse, Purcell_rad_X, color="red", label="X orientation", lw=2)
ax.plot(fUse, Purcell_rad_Y, color="blue", label="Y orientation", lw=2)
ax.plot(fUse, Purcell_rad_Z, color="green", label="Z orientation", lw=2)

# ax.set_yscale("log")
# ax.legend(frameon=True, fontsize=36)
ax.set_ylabel("$\Gamma^\prime/\Gamma_0$")
ax.set_xlabel("$\\nu$ (THz)")

fig.savefig("results/Purcell_nb_rad.pdf")
# %%

# %%

Purcell_1D_X = Purcell_X - Purcell_rad_X
Purcell_1D_Y = Purcell_Y - Purcell_rad_Y
Purcell_1D_Z = Purcell_Z - Purcell_rad_Z


beta_factor_X = Purcell_1D_X / Purcell_X
beta_factor_Y = Purcell_1D_Y / Purcell_Y
beta_factor_Z = Purcell_1D_Z / Purcell_Z


fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(fUse, beta_factor_X, color="red", label="X orientation")
ax.plot(fUse, beta_factor_Y, color="blue", label="Y orientation")
ax.plot(fUse, beta_factor_Z, color="green", label="Z orientation")


# ax.set_yscale("log")

ax.set_ylabel("Beta Factor")
ax.set_xlabel("Frequency (THz)")
ax.legend()
print("Max. beta factor X: ", np.max(beta_factor_X))
print("Max. beta factor Y: ", np.max(beta_factor_Y))


# %%

Gamma_1D_vs_prime_X = Purcell_1D_X / Purcell_rad_X
Gamma_1D_vs_prime_Y = Purcell_1D_Y / Purcell_rad_Y
Gamma_1D_vs_prime_Z = Purcell_1D_Z / Purcell_rad_Z


fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(fUse, Gamma_1D_vs_prime_X, color="red", lw=2, label="X orientation")
ax.plot(fUse, Gamma_1D_vs_prime_Y, color="blue", lw=2, label="Y orientation")
ax.plot(fUse, Gamma_1D_vs_prime_Z, color="green", lw=2, label="Z orientation")

# ax.set_yscale("log")

ax.set_ylabel("$\Gamma_{1D}/\Gamma^\prime$")
ax.set_xlabel("$\\nu$ (THz)")
ax.set_ylim([0, 25])
# ax.legend(frameon=True, fontsize=36)

fig.savefig("results/Gamma_1D_Gamma_prime_nb.pdf")

# %%

Green_function_X = np.load("results/Green_functions_nb_X.npy")
Green_function_Y = np.load("results/Green_functions_nb_Y.npy")
Green_function_Z = np.load("results/Green_functions_nb_Z.npy")

G_Y = Green_function_Y[:, :, 0, :, 1]  # we take the Y component
print(np.shape(G_Y))

# %%
wSimVolX = 2 * 16e-6
discretization_X = np.shape(G_Y)[0]

a = 335e-9

distance_per_dis_u = wSimVolX / discretization_X

print("Distance per discretization unit: ", distance_per_dis_u)
print("Discretization units per unit cell: ", a / distance_per_dis_u)
print("Number of total unit cells:", wSimVolX / a)

# %%

plt.plot(np.imag(G_Y[0:400, np.shape(G_Y)[1] // 2, 125]))

# %%
plt.contourf(np.imag(G_Y[:, :, 100]))

# %%

# Try to obtain the 1D Green's function
# by using the one at the ends and interpolating
# to the centers!

# Then substract this field at the last unit cell
# to get the radiation Green's function
