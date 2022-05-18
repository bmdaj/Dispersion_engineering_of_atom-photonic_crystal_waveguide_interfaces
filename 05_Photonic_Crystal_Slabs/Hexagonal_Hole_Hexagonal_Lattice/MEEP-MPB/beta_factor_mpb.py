# %%
import meep as mp
from meep import mpb

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})
plt.rcParams.update(
    {"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}
)
# for Palatino and other serif fonts use:
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    }
)

import sys

sys.path.append("/home/ben/Desktop/Thesis/github/Thesis_name/Simulations/")
from ExternalFunctions import (
    calculate_ifield,
    plot_field_2D_YZ,
    plot_field_2D_XZ,
    plot_field_2D_XY,
    plot_intentsity_in_z_axis,
    geom_hexagon_PCS,
    plot_unit_cell_cross_sections,
    plot_epsilon_XY,
)

# %%


def get_e_h_fields(sim_params, num_period, custom_material=False):

    geometry_lattice, k_point, geometry, resolution, num_bands = sim_params

    if custom_material == True:

        ms = mpb.ModeSolver(
            geometry_lattice=geometry_lattice,
            k_points=[k_point],
            default_material=geometry,
            resolution=resolution,
            num_bands=num_bands,
        )

    else:

        ms = mpb.ModeSolver(
            geometry_lattice=geometry_lattice,
            k_points=[k_point],
            geometry=geometry,
            resolution=resolution,
            num_bands=num_bands,
        )

    efields = []
    hfields = []

    def get_efields(ms, band):
        efields.append(ms.get_efield(band, bloch_phase=True))

    def get_hfields(ms, band):
        hfields.append(ms.get_hfield(band, bloch_phase=True))

    ms.run(mpb.output_at_kpoint(k_point, mpb.fix_efield_phase, get_efields))

    # Create an MPBData instance to transform the dfields
    md = mpb.MPBData(rectify=True, resolution=resolution, periods=num_period)

    converted_e_x = []
    converted_e_y = []
    converted_e_z = []

    for f in efields:
        # Get just the components of the electric fields and calculate intensity
        efieldx = md.convert(f[..., 0])
        efieldy = md.convert(f[..., 1])
        efieldz = md.convert(f[..., 2])

        converted_e_x.append(efieldx)
        converted_e_y.append(efieldy)
        converted_e_z.append(efieldz)

    efield = np.array([converted_e_x, converted_e_y, converted_e_z])

    ms.run(mpb.output_at_kpoint(k_point, mpb.fix_hfield_phase, get_hfields))

    # Create an MPBData instance to transform the dfields
    md = mpb.MPBData(rectify=True, resolution=resolution, periods=num_period)

    converted_h_x = []
    converted_h_y = []
    converted_h_z = []

    for f in hfields:
        # Get just the components of the electric fields and calculate intensity
        hfieldx = md.convert(f[..., 0])
        hfieldy = md.convert(f[..., 1])
        hfieldz = md.convert(f[..., 2])

        converted_h_x.append(hfieldx)
        converted_h_y.append(hfieldy)
        converted_h_z.append(hfieldz)

    hfield = np.array([converted_h_x, converted_h_y, converted_h_z])

    eps = ms.get_epsilon()
    converted_eps = md.convert(eps).T

    return efield, hfield, converted_eps


def calculate_PF_wg(efield, hfield, freq):

    omega = 2 * np.pi * freq
    eps_0 = 1
    c = 1
    n = 2  # refractive index of med.
    a = 405e-9
    # Z=0 plane and band 0
    efield_calc = efield[:, :, :, :, 720]
    hfield_calc = hfield[:, :, :, :, 720]

    # For X coordinate

    integral = np.sum(
        np.real(
            efield_calc[:, :, 70:96, 77:115]
            * np.conjugate(hfield_calc[:, :, 70:96, 77:115])
        )
    )

    PF_wg = (
        6
        * np.pi ** 2
        * c ** 5
        * eps_0
        * np.abs(efield_calc[0, 0, :, :]) ** 2
        / (a * n * omega ** 2 * integral)
    )

    return PF_wg


# %%

resolution = 32
num_bands = 1

a = 405  # nm
t = 180 / a
scaling = (1 - t) * 0.5

sx = 1
sy = 1
sz = 15
cell = mp.Vector3(sx, sy, sz)

parameters = [a, t, scaling]
geometry, geometry_lattice = geom_hexagon_PCS(parameters, cell)

sim_params = resolution, cell, geometry
fig, ax = plot_unit_cell_cross_sections(a, sim_params)

# %%

num_period = 3
k_point = mp.Vector3(2 / 3, 1 / 3, 0.0)  # K point

sim_params = geometry_lattice, k_point, geometry, resolution, num_bands

field, converted_eps = calculate_ifield(sim_params, num_period)

# %%

num_band = 0

eps = converted_eps.T
a = 290  # nanometers


ylim = [-375, 375]
zlim = [-375, 375]
title = "YZ projection of I field"

plot_field_2D_YZ(eps, np.array(field), a, resolution, num_band, ylim, zlim, title)

# %%

eps = converted_eps.T
a = 290  # nanometers


xlim = [-375, 375]
ylim = [-375, 375]
title = "XY projection of I field"

plot_field_2D_XY(eps, np.array(field), a, resolution, num_band, xlim, ylim, title)

# %%

eps = converted_eps.T
a = 290  # nanometers


xlim = [-375, 375]
zlim = [-375, 375]
title = "XZ projection of I field"

plot_field_2D_XZ(eps, np.array(field), a, resolution, num_band, xlim, zlim, title)
# %%

num_period = 3
k_point = mp.Vector3(2 / 3, 1 / 3, 0.0)  # K point

sim_params = geometry_lattice, k_point, geometry, resolution, num_bands

efield, hfield, converted_eps = get_e_h_fields(sim_params, num_period)
# %%

c = 1
a = 405e-9

freq = 300e12 * a / c
PF_wg = calculate_PF_wg(efield, hfield, freq)

plt.imshow(PF_wg)

# %%
