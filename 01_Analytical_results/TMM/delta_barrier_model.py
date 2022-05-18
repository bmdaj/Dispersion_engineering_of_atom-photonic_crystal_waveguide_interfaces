# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.style.use("science")
matplotlib.rcParams.update({"font.size": 22})


import sys

sys.path.append(".")

from TMM_class import TMM

# %%

a = 380e-9
na = 1
La = a / 2
N_cells = 1  # 20 unit cells in one axis
freqs = np.linspace(000, 1300, 1000) * 1e12  # frequencies we want to scan
xi = 0.0

TMM_phc = TMM(
    freqs=freqs,
    na=na,
    La=La,
    a=a,
    xi=xi,
    type="Delta",
)

# %%

beta = TMM_phc.dispersion_relation()

# %%

# BAND EDGE CALC:

omega_pos = 1.0

omega_neg = np.arccos((xi ** 2 - 1) / (xi ** 2 + 1)) / np.pi

print(omega_pos, omega_neg)

# %%
# TO BE CHECKED: UNITS ARE NOT NORMALIZED

matplotlib.rcParams.update({"font.size": 28})


fig, ax = plt.subplots(1, 2, figsize=(16, 7))

ax[0].plot(beta * a / np.pi, freqs * a / 3e8, label="Bands")
ax[1].plot(np.abs(np.imag(beta)) * a / np.pi, freqs * a / 3e8)

ax[0].fill_between(
    np.linspace(0, 1, len(beta)),
    np.ones(len(np.real(beta))) * omega_neg,
    np.ones(len(np.real(beta))) * omega_pos,
    alpha=0.7,
    color="yellow",
    label="Band-gap",
)

ax[0].set_ylabel("Re$\{\\omega\\pi c / a\}$")
ax[0].set_xlabel("$\\beta(\\pi/a)$")
ax[1].set_ylabel("Im$\{\\omega\\pi c / a\}$")
ax[1].set_xlabel("$\\beta(\\pi/a)$")
ax[0].legend(loc=4, frameon=True)
fig.savefig("QWS_bands.pdf")
# %%

distances = np.linspace(0, 2.0, 200)

I_pos_band_edge = 1 + np.cos(2 * omega_pos * np.pi * distances)
I_neg_band_edge = 1 - np.cos(2 * omega_neg * np.pi * distances)

# %%


matplotlib.rcParams.update({"font.size": 28})


fig, ax = plt.subplots(figsize=(16, 8))

rect1 = matplotlib.patches.Rectangle(
    (0.5 - 0.025, 0), 0.05, 2, color="gray", alpha=0.5, label="Scatterers"
)
rect2 = matplotlib.patches.Rectangle((1.5 - 0.025, 0), 0.05, 2, color="gray", alpha=0.5)
rect3 = matplotlib.patches.Rectangle((2.5 - 0.025, 0), 0.05, 2, color="gray", alpha=0.5)

ax.add_patch(rect1)
ax.add_patch(rect2)
# ax.add_patch(rect3)


ax.plot(distances, I_pos_band_edge, label="$|E_{\\text{BE},+}(x)|^2$", linewidth=3)
ax.plot(distances, I_neg_band_edge, label="$|E_{\\text{BE},-}(x)|^2$", linewidth=3)
ax.set_xlabel("$x/a$")
ax.set_xlim((distances[0], distances[-1]))
ax.set_ylabel("$|E(x)|^2$")
ax.legend(frameon=True, loc=4)


fig.savefig("mode_profiles_TMM.pdf")
# %%

# FOR A N UNIT CELL 1D PHOTONIC CRYSTAL
xi = 0.2
N = 180
c = 3e8
a = 380e-9
freq = np.linspace(600, 1100, 100000) * 1e12
k = 2 * np.pi * freq / c
TMM_phc = TMM(
    freqs=freq,
    na=na,
    La=La,
    a=a,
    xi=xi,
    type="Delta",
)
beta = np.real(TMM_phc.dispersion_relation())
v_g = c * np.sin(beta * a) / (np.sin(k * a) + xi * np.cos(k * a))
# v_g = np.diff(freq) / np.diff(beta)
n_g = c / v_g
f = np.sqrt(n_g ** 2 - 1) / (n_g + 1)  # np.sign(n_g) * np.sqrt((n_g-1)/(n_g+1))

beta = TMM_phc.dispersion_relation()

t = (1 - f ** 2) / (np.exp(-1j * N * beta * a) - f ** 2 * np.exp(1j * N * beta * a))
r = -(2 * 1j * f * np.sin(N * beta * a)) / (
    np.exp(-1j * N * beta * a) - f ** 2 * np.exp(1j * N * beta * a)
)

transmission = np.abs(t) ** 2
reflection = np.abs(r) ** 2
plt.plot(freq * a / (c), transmission)
plt.plot(freq * a / (c), reflection)

# %%
xi = 0.2
c = 3e8
a = 380e-9
freq = np.linspace(600, 1000, 1000) * 1e12
k = 2 * np.pi * freq / c
TMM_phc = TMM(freqs=freq, na=na, La=La, a=a, xi=xi, N_cells=25, type="Delta")
transmission = TMM_phc.calculate_transmission()
reflection = TMM_phc.calculate_reflection()

matplotlib.rcParams.update({"font.size": 28})
import matplotlib.transforms as mtransforms

# fig, ax = plt.subplots(1,2, figsize=(16, 8))
fig, ax = plt.subplot_mosaic([["a)", "b)"]], figsize=(16, 8), constrained_layout=True)

ax["a)"].plot(freq * a / (c), transmission, label="$T_N$", linewidth=2)
ax["a)"].plot(freq * a / (c), reflection, label="$R_N$", linewidth=2)
ax["a)"].legend(frameon=True)

TMM_phc = TMM(freqs=freq, na=na, La=La, a=a, xi=xi, N_cells=50, type="Delta")
transmission = TMM_phc.calculate_transmission()
reflection = TMM_phc.calculate_reflection()

ax["b)"].plot(freq * a / (c), transmission, label="$T_N$", linewidth=2)
ax["b)"].plot(freq * a / (c), reflection, label="$R_N$", linewidth=2)
ax["b)"].legend(frameon=True)
ax["b)"].set_xlabel("Re$\{\\omega\\pi c / a\}$")
ax["a)"].set_xlabel("Re$\{\\omega\\pi c / a\}$")

# for label, ax1 in ax.items():
# label physical distance to the left and up:
#    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
#    ax1.text(0.0, 1.0, label, transform=ax1.transAxes + trans,
#            fontsize='32', va='bottom', fontfamily='serif')
fig.savefig("transmission_reflection_TMM.pdf")
# %%


def atom_reflection(ratio, detuning):
    chi_0 = 1j * ratio * (1 / (1 - 2j * detuning))
    return chi_0 / (chi_0 - 1j)


def atom_eff_transmission(ratio, detuning):
    Gamma_prime = 1
    Gamma_1D = ratio
    Gamma_tot = 1 + ratio
    Delta = Gamma_prime * detuning
    return 1 - (1 - (Gamma_prime ** 2 / Gamma_tot ** 2)) * (
        1 / (1 + 4 * (detuning / Gamma_tot) ** 2)
    )


fig, ax = plt.subplots(1, 2, figsize=(16, 6))

gamma_1D_gamma_prime_list = [0.01, 0.05, 0.1, 0.2, 0.3]
detuning = np.linspace(-5, 5, 1000, dtype=np.complex128)
r = np.zeros((5, len(detuning)))
colors = ["green", "blue", "purple", "red", "orange"]
string = "$\\gamma_{\\text{1D}}/\Gamma^\\prime=$ "
labels_gamma = [
    string + "0.01",
    string + "0.05",
    string + "0.1",
    string + "0.2",
    string + "0.3",
]
for i in range(5):
    r[i, :] = atom_reflection(gamma_1D_gamma_prime_list[i], detuning)
    R = r * np.conj(r)
    T = np.ones(np.shape(r)) - R
    ax[0].plot(detuning, T[i, :], color=colors[i], linewidth=2, label=labels_gamma[i])


# ax[0].plot(detuning, R, linewidth=2, label="$R$")
ax[0].set_xlabel("$\\Delta/\\Gamma^\\prime$")
ax[0].set_ylabel("T")
ax[0].set_title("Infinite waveguide", pad=14)
ax[1].set_title("Finite waveguide", pad=14)
ax[0].set_xlim(detuning[0], detuning[-1])
ax[1].set_xlim(detuning[0], detuning[-1])
ax[0].set_ylim(0.8, 1)
ax[1].set_ylim(0.0, 1)

# ax[0].legend(frameon=True)

T_eff = np.zeros((5, len(detuning)))
Gamma_1D_Gamma_prime_list = [0.1, 0.5, 1.0, 2.0, 3]
string = "$\\Gamma_{\\text{1D}}/\Gamma^\\prime=$ "
labels_gamma = [
    string + "0.1",
    string + "0.5",
    string + "1",
    string + "2",
    string + "3",
]
for i in range(5):
    T_eff[i, :] = atom_eff_transmission(Gamma_1D_Gamma_prime_list[i], detuning)
    ax[1].plot(
        detuning, T_eff[i, :], color=colors[i], linewidth=2, label=labels_gamma[i]
    )
# T_eff = atom_eff_transmission(Gamma_1D_Gamma_prime, detuning)
# R_eff = 1 - T_eff

# ax[1].plot(detuning, T_eff, linewidth=2, label="$T_{\\text{eff}}$")
# ax[1].plot(detuning, R_eff, linewidth=2, label="$R_{\\text{eff}}$")
ax[1].set_xlabel("$\\Delta/\\Gamma^\\prime$")
# ax[0].legend(bbox_to_anchor=(2.25, 1.05), loc="upper left", frameon=True)
# ax[1].legend(bbox_to_anchor=(1.05, 0.5), loc="upper left", frameon=True)
ax[0].legend(loc="best", frameon=True, fontsize=20)
# %%
fig
fig.savefig("t_r_single_atom.pdf")
# %%

freqs = np.linspace(0.87, 0.875, 1000) * c / a
freqs_bot = freqs
a = 380e-9
na = 1
La = a / 2
N_cells = 100
xi = 0.2
TMM_phc = TMM(freqs=freqs, na=na, La=La, a=a, xi=xi, N_cells=N_cells, type="Delta")
cell_index = np.linspace(1, N_cells, N_cells)
intensity_bot = np.zeros((len(freqs), len(cell_index)))

for j in range(len(cell_index)):
    intensity_bot[:, j] = TMM_phc.calculate_intensity_profile(cell_index[j])

freqs = np.linspace(0.999, 1.005, 1000) * c / a
TMM_phc = TMM(freqs=freqs, na=na, La=La, a=a, xi=xi, N_cells=N_cells, type="Delta")
intensity_top = np.zeros((len(freqs), len(cell_index)))

for j in range(len(cell_index)):
    intensity_top[:, j] = TMM_phc.calculate_intensity_profile(cell_index[j])
# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].imshow(
    np.sqrt(intensity_bot),
    aspect="auto",
    origin="lower",
    extent=[cell_index[0], cell_index[-1], freqs_bot[0] * a / c, freqs_bot[-1] * a / c],
    vmin=0.0,
    vmax=0.75,
    cmap="inferno",
)
ax[0].set_xlabel("Cell index")
ax[0].set_ylabel("Re$\{\\omega\\pi c / a\}$")
ax[0].set_title("Dielectric band edge")
ax[0].text(x=33, y=0.8742, s="Band-gap", color="white", weight="bold")
ax[1].imshow(
    np.sqrt(intensity_top),
    aspect="auto",
    origin="lower",
    extent=[cell_index[0], cell_index[-1], freqs[0] * a / c, freqs[-1] * a / c],
    cmap="inferno",
    vmin=0.0,
    vmax=0.75,
)
ax[1].set_xlabel("Cell index")
# ax[1].set_ylabel("Re$\{\\omega\\pi c / a\}$")
ax[1].set_title("Air band edge")
ax[1].text(x=33, y=0.9995, s="Band-gap", color="white", weight="bold")
fig.savefig("supermodes.pdf")

# %%


def norm_transmission(detuning, J, Gamma):
    nominator = detuning + 0.5j
    denominator = detuning + J + 0.5j * (1 + Gamma)
    t = nominator / denominator
    return t * np.conj(t)


detuning_list = np.linspace(-5, 5, 250)
T_disip = np.zeros((5, 250))
T_disp = np.zeros((5, 250))
# Dissipative spectra

J = 0.0
Gamma_list = [0.1, 0.5, 1.0, 5.0, 10.0]

for i in range(len(Gamma_list)):
    T_disip[i, :] = norm_transmission(detuning_list, J, Gamma_list[i])

# Dispersive spectra
J_list = [0.1, 0.5, 1.0, 2.0, 4.0]
Gamma = 0.0

for i in range(len(J_list)):
    T_disp[i, :] = norm_transmission(detuning_list, J_list[i], Gamma)

# %%
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

colors = ["green", "blue", "purple", "red", "orange"]
string = "$\\Gamma_{\\text{1D}}/\Gamma^\\prime=$ "
labels_Gamma = [
    string + "0.1",
    string + "0.5",
    string + "1",
    string + "5",
    string + "10",
]
for i in range(len(Gamma_list)):
    ax[0].plot(
        detuning_list, T_disip[i], color=colors[i], label=labels_Gamma[i], linewidth=2
    )
ax[0].legend(bbox_to_anchor=(2.25, 1.05), loc="upper left", frameon=True)
ax[0].set_title("Dissipative spectra: $J=0$", pad=15)
ax[0].set_xlim(detuning[0], detuning[-1])
ax[0].set_ylim(0, 1)
string = "$J/\Gamma^\\prime=$ "
labels_J = [string + "0.1", string + "0.5", string + "1", string + "2", string + "4"]
for i in range(len(Gamma_list)):
    ax[1].plot(
        detuning_list, T_disp[i], color=colors[i], label=labels_J[i], linewidth=2
    )
# ax[1].legend(frameon=True)
ax[1].set_title("Dispersive spectra: $\\Gamma_{\\text{1D}}=0$", pad=15)
ax[1].set_ylim(0, 8)
ax[1].set_xlim(detuning[0], detuning[-1])
ax[0].set_ylabel("$T/T_0$")
ax[0].set_xlabel("$\\Delta_A/\\Gamma^\\prime$")
ax[1].set_xlabel("$\\Delta_A/\\Gamma^\\prime$")
ax[1].legend(bbox_to_anchor=(1.05, 0.5), loc="upper left", frameon=True)
fig.savefig("single_atom_transmission_green.pdf")
# %%


def transmission_cavity(ratio, detuning, kxi):
    def J_cav(kxi, ratio):
        Gamma_1D = 1  # units of Gamma_prime
        return Gamma_1D * ratio * np.cos(kxi) ** 2

    def Gamma_cav(kxi):
        Gamma_1D = 1  # units of Gamma_prime
        return Gamma_1D * np.cos(kxi) ** 2

    denominator = detuning + 0.5j
    nominator = detuning + J_cav(kxi, ratio) + 0.5j * (1 + Gamma_cav(kxi))

    return denominator / nominator


ratios = np.array([-1.0, -0.5, 0, 0.5, 1.0])
detuning = np.linspace(-10, 10, 100)
T = np.zeros((len(ratios), len(detuning)))
kxi = 0.0

for i in range(len(ratios)):
    t = transmission_cavity(ratios[i], detuning, kxi)
    T[i, :] = np.real(t * np.conj(t))
# %%

fig, ax = plt.subplots(figsize=(14, 8))

colors = ["green", "blue", "purple", "red", "orange"]
string = "$\\Delta_c/\\kappa=$ "
labels = [
    string + "-1.0",
    string + "-0.5",
    string + "0",
    string + "0.5",
    string + "1.0",
]

for i in range(len(ratios)):
    ax.plot(detuning, T[i, :], color=colors[i], label=labels[i], lw=2)

ax.set_xlim(detuning[0], detuning[-1])
ax.set_ylim(0, 2.25)
ax.legend(loc=1, frameon=True)
ax.set_xlabel("$\\Delta_A/\\Gamma^\\prime$")
ax.set_ylabel("$T/T_0$")
fig.savefig("transmission_cavity.pdf")
# %%
