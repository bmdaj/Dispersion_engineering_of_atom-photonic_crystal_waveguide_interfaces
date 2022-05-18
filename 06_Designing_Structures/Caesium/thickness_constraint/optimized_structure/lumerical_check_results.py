#%%
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams.update({"font.size": 24})

#%%

Purcell_X = np.loadtxt("results/PurcellEnhancementX.txt")


freq = 351.726e12
bandwidth = 250e12  # This is giant. will it work?
freqhighres = 1000

# For use as input into the DipolePower() and SourcePower()
fUse = (
    freq
    - 0.5 * bandwidth
    + bandwidth * np.linspace(0, (freqhighres - 1), freqhighres) / (freqhighres - 1)
) * 10 ** -12

fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(fUse, Purcell_X, color="red", label="X orientation")

ax.set_yscale("log")

ax.set_ylabel("Purcell Factor")
ax.set_xlabel("Frequency (THz)")

ax.legend()

print("Max. Purcell Factor in X: ", np.max(Purcell_X))
print("Min. Purcell Factor in X: ", np.min(Purcell_X))

fig.savefig("results/phc_purcell_factor.pdf")

# %%


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


value = find_nearest(fUse, freq * 1e-12)
index = list(fUse).index(value)

Purcell_rad_X = np.loadtxt("results/Purcell_radX.txt")


fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(fUse, Purcell_rad_X, color="red", label="X orientation")

# ax.set_yscale("log")
ax.legend(loc=0)
ax.set_ylabel("$\Gamma^\prime / \Gamma_0$")
ax.set_xlabel("Frequency (THz)")

print("Rad. emission in X for trans. freq. : ", Purcell_rad_X[index])

fig.savefig("results/phc_gamma_prime_gamma_0.pdf")

# %%

Purcell_1D_X = Purcell_X - Purcell_rad_X
beta_factor_X = Purcell_1D_X / Purcell_X


fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(fUse, beta_factor_X, color="red", label="X orientation")
# ax.plot(fUse, beta_factor_Z, color="green", label="Z orientation")


# ax.set_yscale("log")

ax.set_ylabel("Beta Factor")
ax.set_xlabel("Frequency (THz)")
ax.legend()
print("Max. beta factor X: ", np.max(beta_factor_X))


# %%

Gamma_1D_vs_prime_X = Purcell_1D_X / Purcell_rad_X


fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(fUse, Gamma_1D_vs_prime_X, color="red", label="X orientation")

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
print("Min. Gamma2D/Gamma' in X: ", np.min(np.abs(Gamma_1D_vs_prime_X)))

fig.savefig("results/phc_gamma_2D_gamma_prime.pdf")

# %%
