import numpy as np
import matplotlib.pyplot as plt


def create_matrices(freqs, na, nb, La, Lb):
    def interface_matrix(na, nb):

        I = np.zeros((1000, 2, 2))

        I[:, 0, 0] = 0.5 * (na + nb) / (na)
        I[:, 1, 1] = I[:, 0, 0]
        I[:, 0, 1] = 0.5 * (na - nb) / (na)
        I[:, 1, 0] = I[:, 0, 1]

        return I

    def isotropic_medium_matrix(n, k, L):

        P = np.zeros((1000, 2, 2), dtype=complex)

        P[:, 0, 0] = np.exp(1j * n * k * L)
        P[:, 1, 1] = np.exp(-1j * n * k * L)

        return P

    k = 2 * np.pi * freqs / 3e8

    Pa = isotropic_medium_matrix(na, k, 0.5 * La)  # Half layer for first medium
    Pb = isotropic_medium_matrix(nb, k, Lb)  # Layer for the second medium
    Iab = interface_matrix(na=nb, nb=na)  # Interface from first to second medium
    Iba = interface_matrix(na=na, nb=nb)  # Interface from second to first medium

    return Pa, Pb, Iab, Iba


na = 1
nb = 2
a = 380e-9
c = 3e8

La = a / 2
Lb = La

N_cells = 100


cell_index = np.linspace(0, N_cells - 1, N_cells)
freqs = np.linspace(0.2, 0.8, 1000) * c / a
freqs_bot = freqs
intensity_bot = np.zeros((len(freqs), len(cell_index)))

Pa, Pb, Iab, Iba = create_matrices(na=na, nb=nb, La=La, Lb=Lb, freqs=freqs)

U_unit_cell = Pa @ Iab @ Pb @ Iba @ Pa  # Matrix for one unit cell

unit_cell = U_unit_cell

U_total = U_unit_cell

for i in range(N_cells - 1):

    U_total = U_total @ U_unit_cell

L_total = unit_cell

for j in range(len(cell_index)):
    for _ in range(j + 1):
        L_total = L_total @ unit_cell

        term_1 = (1 / L_total[:, 0, 0]) / (U_total[:, 1, 1])
        term_2 = (1 / L_total[:, 1, 0]) / (U_total[:, 1, 1])

        I = term_1 * np.conj(term_1) + term_2 * np.conj(term_2)

    intensity_bot[:, j] = np.real(I)

freqs = np.linspace(0.9995, 1.0005, 1000) * c / a
freqs_top = freqs
intensity_top = np.zeros((len(freqs), len(cell_index)))

Pa, Pb, Iab, Iba = create_matrices(na=na, nb=nb, La=La, Lb=Lb, freqs=freqs)

U_unit_cell = Pa @ Iab @ Pb @ Iba @ Pa  # Matrix for one unit cell

unit_cell = U_unit_cell

U_total = U_unit_cell

for i in range(N_cells - 1):

    U_total = U_total @ U_unit_cell

L_total = unit_cell

for j in range(len(cell_index)):
    for _ in range(j + 1):
        L_total = L_total @ unit_cell

        term_1 = (1 / L_total[:, 0, 0]) / (U_total[:, 1, 1])
        term_2 = (1 / L_total[:, 1, 0]) / (U_total[:, 1, 1])

        I = term_1 * np.conj(term_1) + term_2 * np.conj(term_2)

    intensity_top[:, j] = np.real(I)

# %%

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].imshow(
    np.sqrt(intensity_bot),
    aspect="auto",
    origin="lower",
    extent=[cell_index[0], cell_index[-1], freqs_bot[0] * a / c, freqs_bot[-1] * a / c],
    cmap="inferno",
    vmax=10,
)
ax[0].set_xlabel("Cell index")
ax[0].set_ylabel("Re$\{\\omega\\pi c / a\}$")
ax[0].set_title("Dielectric band edge")
# ax[0].text(x=33, y=0.8742, s="Band-gap", color="white", weight="bold")
ax[1].imshow(
    np.sqrt(intensity_top),
    aspect="auto",
    origin="lower",
    extent=[cell_index[0], cell_index[-1], freqs[0] * a / c, freqs[-1] * a / c],
    cmap="inferno",
    vmax=1000,
)
ax[1].set_xlabel("Cell index")
# ax[1].set_ylabel("Re$\{\\omega\\pi c / a\}$")
ax[1].set_title("Air band edge")
# ax[1].text(x=33, y=0.9995, s="Band-gap", color="white", weight="bold")
# fig.savefig("supermodes.pdf")

# %%
