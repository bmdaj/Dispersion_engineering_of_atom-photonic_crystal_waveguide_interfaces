import numpy as np
from scipy.signal import argrelextrema
from scipy.optimize import fsolve


class TMM:
    def __init__(
        self,
        freqs,
        na,
        La,
        type,
        N_cells=1,
        N_taper=0,
        a=0.0,
        nb=1.0,
        Lb=0.0,
        xi=0.0,
    ):

        self.N_cells = N_cells
        self.N_taper = N_taper
        self.a = a
        self.na = na
        self.La = La
        self.nb = nb
        self.Lb = Lb
        self.xi = xi
        self.freqs = freqs
        self.type = type
        self.unit_cell = None

        self.create_matrix_structure()

    def isotropic_medium_matrix(self, n, k, L):

        P = np.zeros((len(self.freqs), 2, 2), dtype=complex)

        P[:, 0, 0] = np.exp(1j * n * k * L)
        P[:, 1, 1] = np.exp(-1j * n * k * L)

        return P

    def delta_barrier_matrix(self):

        M = np.zeros((len(self.freqs), 2, 2), dtype=complex)

        M[:, 0, 0] = 1 + 1j * self.xi
        M[:, 1, 0] = -1j * self.xi
        M[:, 1, 1] = 1 - 1j * self.xi
        M[:, 0, 1] = +1j * self.xi

        return M

    def interface_matrix(self, na, nb):

        I = np.zeros((len(self.freqs), 2, 2))

        I[:, 0, 0] = 0.5 * (na + nb) / (na)
        I[:, 1, 1] = I[:, 0, 0]
        I[:, 0, 1] = 0.5 * (na - nb) / (na)
        I[:, 1, 0] = I[:, 0, 1]

        return I

    def create_matrices(self, na, nb, La, Lb):

        k = 2 * np.pi * self.freqs / 3e8

        Pa = self.isotropic_medium_matrix(
            na, k, 0.5 * La
        )  # Half layer for first medium
        Pb = self.isotropic_medium_matrix(nb, k, Lb)  # Layer for the second medium
        Iab = self.interface_matrix(
            na=nb, nb=na
        )  # Interface from first to second medium
        Iba = self.interface_matrix(
            na=na, nb=nb
        )  # Interface from second to first medium

        M = self.delta_barrier_matrix()

        return Pa, Pb, Iab, Iba, M

    def add_taper(self, system_matrix):
        # PROBABLY WRONG, NEEDS TO BE DEBUGGED.

        tapering_range = np.linspace(self.na, self.nb, self.N_taper)
        sys_matrix = system_matrix

        for i, index in enumerate(tapering_range):

            Pa, Pb, Iab, Iba = self.create_matrices(
                na=index, nb=self.nb, La=self.La, Lb=self.Lb
            )
            U_unit_cell = Pa @ Iab @ Pb @ Iba @ Pa
            sys_matrix = U_unit_cell @ sys_matrix @ U_unit_cell

        return sys_matrix

    def create_matrix_structure(self):

        Pa, Pb, Iab, Iba, M = self.create_matrices(
            na=self.na, nb=self.nb, La=self.La, Lb=self.Lb
        )
        if self.type == "Periodic":
            U_unit_cell = Pa @ Iab @ Pb @ Iba @ Pa  # Matrix for one unit cell

            self.unit_cell = U_unit_cell

            U_total = U_unit_cell

            for i in range(self.N_cells):

                U_total = U_total @ U_unit_cell

            if self.N_taper > 0:

                U_total = self.add_taper(system_matrix=U_total)

        elif self.type == "Delta":

            U_unit_cell = Pa @ M @ Pa

            self.unit_cell = U_unit_cell

            U_total = U_unit_cell

            for i in range(self.N_cells - 1):

                U_total = U_total @ U_unit_cell

        elif self.type == "Film":

            U_total = Iba @ Pa @ Iab

        else:

            raise ValueError("This geometry type is not supported.")

        self.matrix = U_total

    def calculate_transmission(self):  # to be reviewed
        return np.abs(np.linalg.det(self.matrix) / self.matrix[:, 1, 1]) ** 2

    def calculate_reflection(self):
        return np.abs(self.matrix[:, 1, 0] / self.matrix[:, 1, 1]) ** 2

    def dispersion_relation(self):
        return np.arccos(0.5 * np.trace(self.matrix, axis1=1, axis2=2)) / self.a

    def group_velocity(self):
        w = 2 * np.pi * self.freqs * self.a / 3e8
        beta = np.real(self.dispersion_relation() * self.a)
        v_g = np.diff(np.real(w)) / np.diff(np.real(beta))
        # We will have discontinuities at band-edges when doing the derivative, derivative is not defined at corners:
        # for local maxima
        max_index = argrelextrema(beta, np.greater)
        # for local minima
        min_index = argrelextrema(beta, np.less)

        for index in max_index:
            v_g[index - 1] = v_g[index - 2]
            v_g[index] = v_g[index - 2]
            # v_g[index + 1] = v_g[index - 2]

        for index in min_index:
            v_g[index - 1] = v_g[index - 2]
            v_g[index] = v_g[index - 2]
            # v_g[index + 1] = v_g[index - 2]

        return np.abs(v_g) * self.N_cells

    def group_velocity_v1(self):
        # As they did in paper, does not work...
        L = self.N_cells * self.a

        t = 1 / self.matrix[:, 1, 1]
        t_prime = np.linalg.det(self.matrix) / self.matrix[:, 1, 1]

        z = np.imag(t) / np.real(t)
        z_prime = np.imag(t_prime) / np.real(t_prime)

        return ((1 / L) * (z_prime) / (1 + z ** 2)) ** -1

    def mean_group_velocity(self):
        L = self.N_cells * self.a
        w = 2 * np.pi * self.freqs

        phi = np.angle(1 / self.matrix[:, 1, 1])

        v_g = L * np.diff(np.real(w)) / np.diff(np.real(phi))

        # for local maxima
        max_index = argrelextrema(phi, np.greater)
        # for local minima
        min_index = argrelextrema(phi, np.less)

        for index in max_index:
            v_g[index - 1] = v_g[index - 2]
            v_g[index] = v_g[index - 2]
            v_g[index + 1] = v_g[index - 2]

        for index in min_index:
            v_g[index - 1] = v_g[index - 2]
            v_g[index] = v_g[index - 2]
            v_g[index + 1] = v_g[index - 2]

        return v_g

    def group_velocity_estimation(self):

        L = self.N_cells * self.a
        res_index = argrelextrema(
            self.calculate_reflection(), np.greater
        )  # calculates the index of local maxima, this is, etalon resonances
        res_freq = self.freqs[res_index]
        freq_diff = np.diff(res_freq)
        res_v = L * 2 * freq_diff
        return res_freq[1:], res_v

    def dispersion_model_fit(self, obj):
        # To be done

        if obj == "velocity":

            out = None

        elif obj == "index":

            out = None

        else:
            raise ValueError("This fit type is not supported.")

    def calculate_intensity_profile(self, cell_index):
        # Excited from the right, and only implemented for delta like potentials.

        E_r = 1  # arbitrary units

        if cell_index > self.N_cells:
            raise ValueError("Cell index is higher than number of cells.")

        Pa, Pb, Iab, Iba, M = self.create_matrices(
            na=self.na, nb=self.nb, La=self.La, Lb=self.Lb
        )

        unit_cell = Pa @ M @ Pa
        L_total = unit_cell

        for _ in range(int(cell_index)):
            L_total = L_total @ unit_cell

        term_1 = E_r * (1 / L_total[:, 0, 0]) / (self.matrix[:, 1, 1])
        term_2 = E_r * (1 / L_total[:, 0, 1]) / (self.matrix[:, 1, 1])

        I = np.real(term_1 * np.conj(term_1) + term_2 * np.conj(term_2))

        return I
