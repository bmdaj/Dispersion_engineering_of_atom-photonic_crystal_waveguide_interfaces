#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use("science")
mpl.rcParams.update({"font.size": 46})
#%%

Purcell_X = np.loadtxt("results/PurcellEnhancement_phc_X.txt")
Purcell_Y = np.loadtxt("results/PurcellEnhancement_phc_Y.txt")
Purcell_Z = np.loadtxt("results/PurcellEnhancement_phc_Z.txt")

freq = 351.726e12
bandwidth = 150e12  # This is giant. will it work?
freqhighres = 400

# For use as input into the DipolePower() and SourcePower()
fUse = (
    freq
    - 0.5 * bandwidth
    + bandwidth * np.linspace(0, (freqhighres - 1), freqhighres) / (freqhighres - 1)
) * 10 ** -12

fig, ax = plt.subplots(figsize=(12, 10))

ax.plot(fUse, Purcell_X, color="red", label="X orientation", lw=2)
ax.plot(fUse, Purcell_Y, color="blue", label="Y orientation", lw=2)
# ax.plot(fUse, Purcell_Z, color="green", label="Z orientation")

ax.set_yscale("log")

ax.set_ylabel("PF$\equiv \Gamma_{\\text{tot}} / \Gamma_0$")
ax.set_xlabel("$\\nu$ (THz)")

ax.legend(frameon=True)

print("Max. Purcell Factor in X: ", np.max(Purcell_X))
print("Max. Purcell Factor in Y: ", np.max(Purcell_Y))
print("Min. Purcell Factor in X: ", np.min(Purcell_X))
print("Min. Purcell Factor in Y: ", np.min(Purcell_Y))

fig.savefig("results/phc_purcell_factor_opt_Cs.pdf")

# %%


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


value = find_nearest(fUse, freq * 1e-12)
index = list(fUse).index(value)

Purcell_rad_X = np.loadtxt("results/Purcell_rad_phc_X.txt")
Purcell_rad_Y = np.loadtxt("results/Purcell_rad_phc_Y.txt")
Purcell_rad_Z = np.loadtxt("results/Purcell_rad_phc_Z.txt")


fig, ax = plt.subplots(figsize=(12, 10))

ax.plot(fUse, Purcell_rad_X, color="red", label="X orientation", lw=2)
ax.plot(fUse, Purcell_rad_Y, color="blue", label="Y orientation", lw=2)
# ax.plot(fUse, Purcell_rad_Z, color="green", label="Z orientation")

# ax.set_yscale("log")
# ax.legend(loc=0)
ax.set_ylabel("$\Gamma^\prime / \Gamma_0$")
ax.set_xlabel("$\\nu$ (THz)")

print("Rad. emission in X for trans. freq. : ", Purcell_rad_X[index])
print("Rad. emission in Y for trans. freq ", Purcell_rad_Y[index])
print("Rad. emission min in X: ", np.min(Purcell_rad_X))
print("Rad. emission min in Y: ", np.min(Purcell_rad_Y))

fig.savefig("results/phc_gamma_prime_gamma_0_opt_Cs.pdf")

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
# ax.plot(fUse, beta_factor_Z, color="green", label="Z orientation")


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


fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(fUse, Gamma_1D_vs_prime_X, color="red", label="X orientation")
ax.plot(fUse, Gamma_1D_vs_prime_Y, color="blue", label="Y orientation")
# ax.plot(fUse, Gamma_1D_vs_prime_Z, color="green", label="Z orientation")

ax.set_yscale("log")
# ax.set_xlim(365, 380)

ax.set_ylabel("$\Gamma_{2D}/\Gamma^\prime$")
ax.set_xlabel("Frequency (THz)")
ax.legend()

print("Max. Gamma2D/Gamma' in X: ", np.max(Gamma_1D_vs_prime_X))
print(
    "Frequency for maximum: ",
    fUse[list(Gamma_1D_vs_prime_X).index(np.max(Gamma_1D_vs_prime_X))],
)
print("Max. Gamma2D/Gamma' in Y: ", np.max(Gamma_1D_vs_prime_Y))
print("Min. Gamma2D/Gamma' in X: ", np.min(np.abs(Gamma_1D_vs_prime_X)))
print("Min. Gamma2D/Gamma' in Y: ", np.min(np.abs(Gamma_1D_vs_prime_Y)))

fig.savefig("results/phc_gamma_2D_gamma_prime.pdf")

# %%

freqhighres = 500

fUse = (
    freq
    - 0.5 * bandwidth
    + bandwidth * np.linspace(0, (freqhighres - 1), freqhighres) / (freqhighres - 1)
) * 10 ** -12


Green_function_circ = np.load("results/Green_functions_phc_circ.npy")
index_z = np.load("results/index_phc_circ.npy")


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


value = find_nearest(fUse, 351.73)
index = list(fUse).index(value)

T_1 = (
    1
    / np.sqrt(2)
    * (
        Green_function_circ[:, :, 0, index, 0]
        + 1j * Green_function_circ[:, :, 0, index, 1]
    )
)

# T_1 = np.sqrt(
#     Green_function_circ[:, :, 0, index, 0] * np.conjugate(Green_function_circ[:, :, 0, index, 0])
#     + Green_function_circ[:, :, 0, index, 1] * np.conjugate(Green_function_circ[:, :, 0, index, 1])
#     + Green_function_circ[:, :, 0, index, 2]  * np.conjugate(Green_function_circ[:, :, 0, index, 2])
# )

index_z = index_z[:, :, 0, 0]


# %%
mpl.rcParams.update({"font.size": 38})
middle_index = np.shape(index_z)[0] // 2
index_low = middle_index - 80
index_high = middle_index + 80

fig, ax = plt.subplots(figsize=(14, 8))

wSimVolXY_tot = 20.0e-6
zoom_extent = 20 * 80 / np.shape(index_z)[0]

ax.contourf(
    index_z[index_low:index_high, index_low:index_high].T,
    cmap="binary",
    alpha=0.1,
    extent=[-zoom_extent, zoom_extent, -zoom_extent, zoom_extent],
)
im = ax.imshow(
    np.real(T_1)[index_low:index_high, index_low:index_high].T,
    vmax=2,
    vmin=-2,
    cmap="seismic",
    extent=[-zoom_extent, zoom_extent, -zoom_extent, zoom_extent],
)

cbar = fig.colorbar(im)
cbar.set_ticks([-2, 2])

ax.set_ylabel("y ($\mu m$)")
ax.set_xlabel("x ($\mu m$)")

T_value = np.real(T_1)[index_low:index_high, index_low:index_high].T

x_space = np.linspace(-zoom_extent, zoom_extent, len(T_value))
y_space = np.linspace(-zoom_extent, zoom_extent, len(T_value))

array = np.asarray(x_space)

aphc = 0.38365

value_NN_1_x = aphc * 0.5 * np.sqrt(3)
value_NN_1_y = aphc * 0.5

idx_NN_1_x = (np.abs(array - value_NN_1_x)).argmin()
idx_NN_1_y = (np.abs(array - value_NN_1_y)).argmin()

value_NN_2_x = 2 * aphc * 0.5 * np.sqrt(3)
value_NN_2_y = 2 * aphc * 0.5

idx_NN_2_x = (np.abs(array - value_NN_2_x)).argmin()
idx_NN_2_y = (np.abs(array - value_NN_2_y)).argmin()

value_NN_3_x = 3 * aphc * 0.5 * np.sqrt(3)
value_NN_3_y = 3 * aphc * 0.5

idx_NN_3_x = (np.abs(array - value_NN_3_x)).argmin()
idx_NN_3_y = (np.abs(array - value_NN_3_y)).argmin()

print("Maximum value at center: ", np.max(abs(T_value)))
print("Value at first neighbor: ", T_value[idx_NN_1_x, idx_NN_1_y])
print("Value at second neighbor: ", T_value[idx_NN_2_x, idx_NN_2_y])
print("Value at third neighbor: ", T_value[idx_NN_3_x, idx_NN_3_y])
print(
    "Decay constant: ",
    T_value[idx_NN_3_x, idx_NN_3_y] / T_value[idx_NN_1_x, idx_NN_1_y],
)


fig.savefig("results/real_green_circ_pnas.pdf")

# %%
# WHAT HAPPENS AT THESE CRAZY RESONANCES?
# Frequency of first peak: 366.5 THz
# Frequency of "normal" results: 368.5 THz
# Frequency f second / biggest resonance: 371 THz

Green_function_circ = np.load("results/Green_functions_phc_circ.npy")
index_z = np.load("results/index_phc_circ.npy")

value = find_nearest(fUse, 371)
print(value)
index = list(fUse).index(value)

# T_1 = (
#    1
#    / np.sqrt(2)
#    * (
#        Green_function_circ[:, :, 0, index, 0]
#        + 1j * Green_function_circ[:, :, 0, index, 1]
#    )
# )

T_1 = np.sqrt(
    Green_function_circ[:, :, 0, index, 0]
    * np.conjugate(Green_function_circ[:, :, 0, index, 0])
    + Green_function_circ[:, :, 0, index, 1]
    * np.conjugate(Green_function_circ[:, :, 0, index, 1])
    + Green_function_circ[:, :, 0, index, 2]
    * np.conjugate(Green_function_circ[:, :, 0, index, 2])
)

index_z = index_z[:, :, 0, 0]


# %%

middle_index = np.shape(index_z)[0] // 2
index_low = middle_index - 425
index_high = middle_index + 425

fig, ax = plt.subplots(figsize=(14, 8))

wSimVolXY_tot = 20.0e-6
zoom_extent = 20 * 425 / np.shape(index_z)[0]

ax.contourf(
    index_z[index_low:index_high, index_low:index_high].T,
    cmap="binary",
    alpha=0.1,
    extent=[-zoom_extent, zoom_extent, -zoom_extent, zoom_extent],
)
im = ax.imshow(
    np.real(T_1)[index_low:index_high, index_low:index_high].T,
    cmap="seismic",
    vmax=5,
    vmin=-5,
    extent=[-zoom_extent, zoom_extent, -zoom_extent, zoom_extent],
)
fig.colorbar(im)
ax.set_ylabel("y ($\mu m$)")
ax.set_xlabel("x ($\mu m$)")

# %%

fig, ax = plt.subplots(figsize=(14, 8))

wSimVolXY_tot = 20.0e-6
zoom_extent = 20 * 425 / np.shape(index_z)[0]

ax.contourf(
    index_z[index_low:index_high, index_low:index_high].T,
    cmap="binary",
    alpha=0.1,
    extent=[-zoom_extent, zoom_extent, -zoom_extent, zoom_extent],
)
im = ax.imshow(
    np.imag(T_1)[index_low:index_high, index_low:index_high].T,
    cmap="seismic",
    extent=[-zoom_extent, zoom_extent, -zoom_extent, zoom_extent],
)
fig.colorbar(im)
ax.set_ylabel("y ($\mu m$)")
ax.set_xlabel("x ($\mu m$)")

# %%
# %%
# SURFACE
# WHAT HAPPENS AT THESE CRAZY RESONANCES?
# Frequency of first peak: 366.5 THz
# Frequency of "normal" results: 368.5 THz
# Frequency f second / biggest resonance: 371 THz

Green_function_circ = np.load("results/Green_functions_phc_circ_surf.npy")
index_z = np.load("results/index_phc_circ.npy")

value = find_nearest(fUse, 371)
print(value)
index = list(fUse).index(value)

T_1 = (
    1
    / np.sqrt(2)
    * (
        Green_function_circ[:, :, 0, index, 0]
        + 1j * Green_function_circ[:, :, 0, index, 1]
    )
)

index_z = index_z[:, :, 0, 0]


# %%

middle_index = np.shape(index_z)[0] // 2
index_low = middle_index - 425
index_high = middle_index + 425

fig, ax = plt.subplots(figsize=(14, 8))

wSimVolXY_tot = 20.0e-6
zoom_extent = 20 * 425 / np.shape(index_z)[0]

ax.contourf(
    index_z[index_low:index_high, index_low:index_high].T,
    cmap="binary",
    alpha=0.1,
    extent=[-zoom_extent, zoom_extent, -zoom_extent, zoom_extent],
)
im = ax.imshow(
    np.real(T_1)[index_low:index_high, index_low:index_high].T,
    cmap="seismic",
    vmax=5,
    vmin=-5,
    extent=[-zoom_extent, zoom_extent, -zoom_extent, zoom_extent],
)
fig.colorbar(im)
ax.set_ylabel("y ($\mu m$)")
ax.set_xlabel("x ($\mu m$)")

# %%

fig, ax = plt.subplots(figsize=(14, 8))

wSimVolXY_tot = 20.0e-6
zoom_extent = 20 * 425 / np.shape(index_z)[0]

ax.contourf(
    index_z[index_low:index_high, index_low:index_high].T,
    cmap="binary",
    alpha=0.1,
    extent=[-zoom_extent, zoom_extent, -zoom_extent, zoom_extent],
)
im = ax.imshow(
    np.imag(T_1)[index_low:index_high, index_low:index_high].T,
    cmap="seismic",
    extent=[-zoom_extent, zoom_extent, -zoom_extent, zoom_extent],
)
fig.colorbar(im)
ax.set_ylabel("y ($\mu m$)")
ax.set_xlabel("x ($\mu m$)")


# %%
# WHAT HAPPENS AT THESE CRAZY RESONANCES?
# Frequency of first peak: 366.5 THz
# Frequency of "normal" results: 368.5 THz
# Frequency f second / biggest resonance: 371 THz

Green_function_circ = np.load("results/Green_functions_phc_X_field.npy")
index_z = np.load("results/index_phc_X_field.npy")

value = find_nearest(fUse, 370.87)
print(value)
index = list(fUse).index(value)
# Taking trace???
T_1 = np.sqrt(
    Green_function_circ[:, :, 0, index, 0]
    * np.conjugate(Green_function_circ[:, :, 0, index, 0])
    + Green_function_circ[:, :, 0, index, 1]
    * np.conjugate(Green_function_circ[:, :, 0, index, 1])
    + Green_function_circ[:, :, 0, index, 2]
    * np.conjugate(Green_function_circ[:, :, 0, index, 2])
)


index_z = index_z[:, :, 0, 0]


# %%

middle_index = np.shape(index_z)[0] // 2
index_low = middle_index - 350
index_high = middle_index + 350

fig, ax = plt.subplots(figsize=(14, 8))

wSimVolXY_tot = 16.0e-6
zoom_extent = 16 * 350 / np.shape(index_z)[0]

ax.contourf(
    index_z[index_low:index_high, index_low:index_high].T,
    cmap="binary",
    alpha=0.1,
    extent=[-zoom_extent, zoom_extent, -zoom_extent, zoom_extent],
)
im = ax.imshow(
    np.real(T_1)[index_low:index_high, index_low:index_high].T,
    cmap="seismic",
    vmax=5,
    vmin=-5,
    extent=[-zoom_extent, zoom_extent, -zoom_extent, zoom_extent],
)
fig.colorbar(im)
ax.set_ylabel("y ($\mu m$)")
ax.set_xlabel("x ($\mu m$)")

# %%

fig, ax = plt.subplots(figsize=(14, 8))

wSimVolXY_tot = 16.0e-6
zoom_extent = 16 * 350 / np.shape(index_z)[0]

ax.contourf(
    index_z[index_low:index_high, index_low:index_high].T,
    cmap="binary",
    alpha=0.1,
    extent=[-zoom_extent, zoom_extent, -zoom_extent, zoom_extent],
)
im = ax.imshow(
    np.imag(T_1)[index_low:index_high, index_low:index_high].T,
    cmap="seismic",
    extent=[-zoom_extent, zoom_extent, -zoom_extent, zoom_extent],
)
fig.colorbar(im)
ax.set_ylabel("y ($\mu m$)")
ax.set_xlabel("x ($\mu m$)")

# %%
