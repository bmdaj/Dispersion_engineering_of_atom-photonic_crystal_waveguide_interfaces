# ---------------------------------------------------------------------------------------------#
# IN THIS MODULE WE DEFINE EXTERNAL FUNCTIONS TO MAKE THE REST OF THE CODE MORE MODULAR
# ---------------------------------------------------------------------------------------------#

import numpy as np
from meep import mpb
import meep as mp
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------#
# GEOMETRY FUNCTIONS AND ROUTINES
# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#
# This function describes the geometry of the APCW structure
# ---------------------------------------------------------------------------------------------#


def geom_apcw(params):

    # Extraction of parameters

    dielec_const, period, amp, width, thickness, gap, dephase = params

    def geometry(Vector3):

        array3 = np.array(Vector3)
        x = array3[0]
        y = array3[1]
        z = array3[2]

        if (
            y >= 0.5 * gap
            and amp * np.cos(2 * np.pi * (x) + dephase) + 0.5 * gap + width >= y
            and 0.5 * thickness > np.abs(z)
        ):
            eps = mp.Medium(index=dielec_const)

        elif (
            y <= -0.5 * width
            and -amp * np.cos(2 * np.pi * (x)) - 0.5 * gap - width <= y
            and 0.5 * thickness > np.abs(z)
        ):
            eps = mp.Medium(index=dielec_const)
        else:
            eps = mp.Medium(epsilon=1)

        if width - amp + 0.5 * gap >= np.abs(
            y
        ) >= 0.5 * gap and 0.5 * thickness > np.abs(z):
            eps = mp.Medium(index=dielec_const)

        return eps

    return geometry


# ---------------------------------------------------------------------------------------------#
# This function describes the geometry of the square  lattice  PCS with sloped sidewalls
# ---------------------------------------------------------------------------------------------#


def geom_square_PCS_slope(params):

    # Extraction of parameters

    dielec_const, r, t, alpha = params

    def geometry(Vector3):

        array3 = np.array(Vector3)
        x = array3[0]
        y = array3[1]
        z = array3[2]

        h = t  #  nanometers
        r1 = r  #  upper radius (biggest one)
        r2 = r1 - h * np.sin(alpha)

        x_prime = (
            0.5 * (r1 + r2) + z * (r1 - r2) / h
        )  # this defines the slope of the walls
        y_prime = x_prime  # the slope is the same in both directions (simmetry)

        if x ** 2 + y ** 2 > x_prime ** 2 and np.abs(z) < h * 0.5:

            eps = mp.Medium(index=dielec_const)

        else:

            eps = mp.Medium(index=1)

        return eps

    return geometry


# ---------------------------------------------------------------------------------------------#
# This function describes the geometry of the hexagonal lattice  PCS with hexagonal holes
# ---------------------------------------------------------------------------------------------#


def geom_hexagon_PCS(params, cell):

    # Extraction of parameters. See PNAS paper.

    a, t, scaling = params

    sx = np.array(cell)[0]
    sy = np.array(cell)[1]
    sz = np.array(cell)[2]

    geometry = [
        mp.Block(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(mp.inf, mp.inf, t),
            material=mp.Medium(index=1.9935),
        )
    ]

    # A hexagonal prism defined by six vertices centered on the origin
    # of material crystalline silicon (from the materials library)

    vertices = [
        mp.Vector3(scaling * 2 / np.sqrt(3), 0),
        mp.Vector3(scaling * 1 / np.sqrt(3), scaling * 1),
        mp.Vector3(-scaling * 1 / np.sqrt(3), scaling * 1),
        mp.Vector3(scaling * -2 / np.sqrt(3), 0),
        mp.Vector3(-scaling * 1 / np.sqrt(3), -scaling * 1),
        mp.Vector3(scaling * 1 / np.sqrt(3), -scaling * 1),
    ]

    vertices_new = []

    # we change tthe vertices to match the basis of our lattice

    geometry_lattice = mp.Lattice(
        size=mp.Vector3(1, 1, sz),
        basis1=mp.Vector3(np.sqrt(3) / 2, 0.5, 0.0),
        basis2=mp.Vector3(np.sqrt(3) / 2, -0.5, 0.0),
    )
    vertices_new = []
    for vertice in vertices:
        vertices_new.append(mp.cartesian_to_lattice(vertice, geometry_lattice))

    geometry.append(
        mp.Prism(
            vertices_new, height=t, center=mp.Vector3(), material=mp.Medium(index=1)
        )
    )

    return geometry, geometry_lattice


# ---------------------------------------------------------------------------------------------#
# This function describes the geometry of the hexagonal lattice  PCS with hexagonal holes with slope
# ---------------------------------------------------------------------------------------------#


def geom_hexagon_PCS_slope(params):

    # Extraction of parameters

    dielec_const, t, alpha, geometry_lattice = params

    def geometry(Vector3):

        array3 = np.array(mp.lattice_to_cartesian(Vector3, geometry_lattice))
        y = np.abs(array3[0])
        x = np.abs(array3[1])
        z = array3[2]

        r1 = (1 - t) / 2  #  upper radius (biggest one)
        r2 = r1 - t * np.sin(alpha)

        r = np.abs(
            0.5 * (r1 + r2) + z * (r1 - r2) / (t)
        )  # this defines the slope of the walls
        h = r / np.sqrt(3)  # r tan(30º)

        condition_x_square = np.abs(x) <= r
        condition_y_square = np.abs(y) <= h

        condition_x_triangle = np.abs(x) <= r
        condition_y_triangle = h <= np.abs(y) <= -h / r * np.abs(x) + 2 * h

        condition_z = np.abs(z) <= t * 0.5

        if condition_x_square and condition_y_square and condition_z:

            eps = mp.Medium(index=1)

        elif condition_x_triangle and condition_y_triangle and condition_z:

            eps = mp.Medium(index=1)

        elif condition_z:

            eps = mp.Medium(index=dielec_const)

        else:

            eps = mp.Medium(index=1)

        return eps

    return geometry


# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------#
# CALCULATION FUNCTIONS AND ROUTINES
# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#
# This function calculates the electric field
# ---------------------------------------------------------------------------------------------#


def calculate_efield(sim_params, num_period, component, custom_material=False):

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

    def get_efields(ms, band):
        efields.append(ms.get_efield(band, bloch_phase=True))

    ms.run(mpb.output_at_kpoint(k_point, mpb.fix_efield_phase, get_efields))

    # Create an MPBData instance to transform the dfields
    md = mpb.MPBData(rectify=True, resolution=resolution, periods=num_period)

    converted = []
    for f in tqdm(efields):

        if component == "X":
            f = f[..., 0, 0]

        if component == "Y":
            f = f[..., 0, 1]

        if component == "Z":
            f = f[..., 0, 2]

        converted.append(md.convert(f))

    eps = ms.get_epsilon()
    converted_eps = md.convert(eps).T

    return converted, converted_eps


# ---------------------------------------------------------------------------------------------#
# This function calculates the intensity field
# ---------------------------------------------------------------------------------------------#


def calculate_ifield(sim_params, num_period, custom_material=False):

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

    def get_efields(ms, band):
        efields.append(ms.get_efield(band, bloch_phase=True))

    ms.run_te(mpb.output_at_kpoint(k_point, mpb.fix_efield_phase, get_efields))

    # Create an MPBData instance to transform the dfields
    md = mpb.MPBData(rectify=True, resolution=resolution, periods=num_period)

    converted = []

    for f in efields:
        # Get just the components of the electric fields and calculate intensity
        efieldx = md.convert(f[..., 0])
        efieldy = md.convert(f[..., 1])
        efieldz = md.convert(f[..., 2])

        f1 = (
            efieldx * np.conjugate(efieldx)
            + efieldy * np.conjugate(efieldy)
            + efieldz * np.conjugate(efieldz)
        )
        converted.append(np.real(f1))

    eps = ms.get_epsilon()
    converted_eps = md.convert(eps).T

    return converted, converted_eps


# ---------------------------------------------------------------------------------------------#
# This function gives back the simulation domain length
# ---------------------------------------------------------------------------------------------#


def simulation_domain(a, eps, resolution):

    domain = np.array(np.shape(eps))
    distance_conversion = a / resolution
    domain_limits = domain * distance_conversion

    print("The simulation domain lengths are:", domain_limits.astype(int))

    return domain_limits.astype(int)


# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------#
# PLOTTING FUNCTIONS AND ROUTINES
# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#
# This function plots the band diagram in THZ
# ---------------------------------------------------------------------------------------------#


def plot_bands_1D_THZ(a, num_bands, files, xlims, ylims):

    from scipy.interpolate import make_interp_spline, BSpline

    fig, ax = plt.subplots(figsize=(14, 10))

    light_speed = 3e8

    f1, f2 = files

    for i in range(1, num_bands):
        xnew = np.linspace((f1[1:, 1] * 2).min(), (f1[1:, 1] * 2).max(), 300)
        spl1 = make_interp_spline(
            f1[1:, 1] * 2, f1[1:, -i] * light_speed / (a * 10 ** 12), k=3
        )  # type: BSpline
        spl2 = make_interp_spline(
            f1[1:, 1] * 2, f2[1:, -i] * light_speed / (a * 10 ** 12), k=3
        )  # type: BSpline
        TE_mode_smooth = spl1(xnew)
        TM_mode_smooth = spl2(xnew)
        if i == 1:
            ax.plot(
                xnew, TE_mode_smooth, c="orange", linestyle="dashed", label="TE modes"
            )
            ax.plot(
                xnew, TM_mode_smooth, c="blue", linestyle="dashed", label="TM modes"
            )
        else:
            ax.plot(xnew, TE_mode_smooth, c="orange", linestyle="dashed")
            ax.plot(xnew, TM_mode_smooth, c="blue", linestyle="dashed")
    ax.fill_between(
        xnew,
        np.ones(np.shape(xnew)) * f1[-1, -1] * light_speed / (a * 10 ** 12),
        np.ones(np.shape(xnew)) * f1[-1, -2] * light_speed / (a * 10 ** 12),
        color="orange",
        alpha=0.2,
        label="TE gap",
    )
    ax.fill_between(
        xnew,
        np.ones(np.shape(xnew)) * f2[-1, -1] * light_speed / (a * 10 ** 12),
        np.ones(np.shape(xnew)) * f2[-1, -2] * light_speed / (a * 10 ** 12),
        color="blue",
        alpha=0.2,
        label="TM gap",
    )
    ax.plot(
        np.linspace(0, 1, 100),
        np.linspace(0, 1, 100) * light_speed / (2 * a * 10 ** 12),
        c="r",
        label="Light line",
    )
    ax.fill_between(
        np.linspace(0, 1, 100),
        np.linspace(0, 1, 100) * light_speed / (2 * a * 10 ** 12),
        1 * np.ones(100) * light_speed / (2 * a * 10 ** 12),
        color="purple",
        alpha=0.8,
        label="Light cone",
    )

    ax.set_title("Dispersion diagram for the APCW waveguide model")
    ax.set_xlabel("Normalized $k_x a / \\pi$")
    ax.set_ylabel("Frequency $\\nu$ (THz)")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.legend()
    return fig


# ---------------------------------------------------------------------------------------------#
# This function plots the dielectric material distribution in YZ plane
# ---------------------------------------------------------------------------------------------#


def plot_epsilon_YZ(
    sim_params, a, num_period, ylim, zlim, title, custom_material=False
):

    geometry_lattice, k_point, default_material, resolution, num_bands = sim_params

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
    ms.run()

    eps = ms.get_epsilon()

    md = mpb.MPBData(rectify=True, periods=num_period, resolution=resolution)

    converted_eps = md.convert(eps).T

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title)

    x_half_index = int(np.shape(converted_eps)[0] // 2)

    distance_conversion = resolution / a

    ymin, ymax = ylim
    zmin, zmax = zlim

    y_offset = int(np.array(np.shape(converted_eps))[1] // 2)
    z_offset = int(np.array(np.shape(converted_eps))[2] // 2)

    ymin_index = int(ymin * distance_conversion) + y_offset
    ymax_index = int(ymax * distance_conversion) + y_offset
    zmin_index = int(zmin * distance_conversion) + z_offset
    zmax_index = int(zmax * distance_conversion) + z_offset

    extent = [zmin, zmax, ymin, ymax]

    ax.imshow(
        converted_eps[x_half_index, ymin_index:ymax_index, zmin_index:zmax_index],
        extent=extent,
        interpolation="spline36",
        cmap="binary",
    )
    ax.set_ylabel("Y distance")
    ax.set_xlabel("Z distance")
    return fig, ax


# ---------------------------------------------------------------------------------------------#
# This function plots the dielectric material distribution in XY plane
# ---------------------------------------------------------------------------------------------#


def plot_epsilon_XY(
    sim_params, a, num_period, xlim, ylim, title, custom_material=False
):

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

    ms.run()

    eps = ms.get_epsilon()

    md = mpb.MPBData(rectify=True, periods=num_period, resolution=resolution)

    converted_eps = md.convert(eps)[
        :,
        :,
    ]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title)

    z_half_index = int(np.shape(converted_eps)[2] // 2)

    distance_conversion = resolution / a

    xmin, xmax = xlim
    ymin, ymax = ylim

    x_offset = int(np.array(np.shape(converted_eps))[0] // 2)
    y_offset = int(np.array(np.shape(converted_eps))[1] // 2)

    xmin_index = int(xmin * distance_conversion) + x_offset
    xmax_index = int(xmax * distance_conversion) + x_offset
    ymin_index = int(ymin * distance_conversion) + y_offset
    ymax_index = int(ymax * distance_conversion) + y_offset

    extent = [xmin, xmax, ymin, ymax]

    ax.imshow(
        converted_eps[xmin_index:xmax_index, ymin_index:ymax_index, z_half_index].T,
        extent=extent,
        interpolation="spline36",
        cmap="binary",
    )
    ax.set_ylabel("Y distance")
    ax.set_xlabel("X distance")
    ax.grid(True, alpha=0.5)
    return fig, ax


# ---------------------------------------------------------------------------------------------#
# This function plots a 2D field in the YZ projection
# ---------------------------------------------------------------------------------------------#


def plot_field_2D_YZ(eps, field, a, resolution, num_band, ylim, zlim, title):

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title)

    x_half_index = int(np.shape(eps)[0] // 2)
    distance_conversion = resolution / a

    ymin, ymax = ylim
    zmin, zmax = zlim

    y_offset = int(np.array(np.shape(eps))[1] // 2)
    z_offset = int(np.array(np.shape(eps))[2] // 2)

    ymin_index = int(ymin * distance_conversion) + y_offset
    ymax_index = int(ymax * distance_conversion) + y_offset
    zmin_index = int(zmin * distance_conversion) + z_offset
    zmax_index = int(zmax * distance_conversion) + z_offset

    extent = [zmin, zmax, ymin, ymax]

    y_range = np.linspace(ymin, ymax, ymax_index - ymin_index)
    z_range = np.linspace(zmin, zmax, zmax_index - zmin_index)

    if len(np.shape(field)) == 4:

        field = np.array(field)[:, x_half_index, :, :]

    ax.contour(
        eps[x_half_index, ymin_index:ymax_index, zmin_index:zmax_index],
        extent=extent,
        cmap="binary",
    )

    posd1 = ax.imshow(
        np.real(field[num_band][ymin_index:ymax_index, zmin_index:zmax_index]),
        extent=extent,
        interpolation="spline36",
        cmap="plasma",
    )
    ax.set_ylabel("Y distance (nm)")
    ax.set_xlabel("Z distance (nm)")
    fig.colorbar(posd1, ax=ax)

    return fig, ax


# ---------------------------------------------------------------------------------------------#
# This function plots a 2D field in the XY projection
# ---------------------------------------------------------------------------------------------#


def plot_field_2D_XY(eps, field, a, resolution, num_band, xlim, ylim, title):

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title)

    z_half_index = int(np.shape(eps)[2] // 2)
    distance_conversion = resolution / a

    xmin, xmax = xlim
    ymin, ymax = ylim

    x_offset = int(np.array(np.shape(eps))[0] // 2)
    y_offset = int(np.array(np.shape(eps))[1] // 2)

    xmin_index = int(xmin * distance_conversion) + x_offset
    xmax_index = int(xmax * distance_conversion) + x_offset
    ymin_index = int(ymin * distance_conversion) + y_offset
    ymax_index = int(ymax * distance_conversion) + y_offset

    print(xmin_index, xmax_index, ymin_index, ymax_index)

    extent = [xmin, xmax, ymin, ymax]

    x_range = np.linspace(xmin, xmax, xmax_index - xmin_index)
    y_range = np.linspace(ymin, ymax, ymax_index - ymin_index)

    if len(np.shape(field)) == 4:

        field = np.array(field)[:, :, :, z_half_index]

    ax.contour(
        eps[xmin_index:xmax_index, ymin_index:ymax_index, z_half_index],
        extent=extent,
        cmap="binary",
    )

    posd1 = ax.imshow(
        np.real(field[num_band][xmin_index:xmax_index, ymin_index:ymax_index]),
        extent=extent,
        interpolation="spline36",
        cmap="plasma",
    )

    ax.set_xlabel("X distance (nm)")
    ax.set_ylabel("Y distance (nm)")

    fig.colorbar(posd1, ax=ax)

    return fig, ax


# ---------------------------------------------------------------------------------------------#
# This function plots a 2D field in the XZ projection
# ---------------------------------------------------------------------------------------------#


def plot_field_2D_XZ(eps, field, a, resolution, num_band, xlim, zlim, title):

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title)

    y_half_index = int(np.shape(eps)[1] // 2)
    distance_conversion = resolution / a

    xmin, xmax = xlim
    zmin, zmax = zlim

    x_offset = int(np.array(np.shape(eps))[0] // 2)
    z_offset = int(np.array(np.shape(eps))[2] // 2)

    xmin_index = int(xmin * distance_conversion) + x_offset
    xmax_index = int(xmax * distance_conversion) + x_offset
    zmin_index = int(zmin * distance_conversion) + z_offset
    zmax_index = int(zmax * distance_conversion) + z_offset

    extent = [zmin, zmax, xmin, xmax]

    x_range = np.linspace(xmin, xmax, xmax_index - xmin_index)
    z_range = np.linspace(zmin, zmax, zmax_index - zmin_index)

    if len(np.shape(field)) == 4:

        field = np.array(field)[:, :, y_half_index, :]

    ax.contour(
        eps[xmin_index:xmax_index, y_half_index, zmin_index:zmax_index],
        extent=extent,
        cmap="binary",
    )

    posd1 = ax.imshow(
        np.real(field[num_band][xmin_index:xmax_index, zmin_index:zmax_index]),
        extent=extent,
        interpolation="spline36",
        cmap="plasma",
    )

    ax.set_ylabel("X distance (nm)")
    ax.set_xlabel("Z distance (nm)")

    fig.colorbar(posd1, ax=ax)

    return fig, ax


# ---------------------------------------------------------------------------------------------#
# This function plots a cross section of the unit cell in a PCS
# ---------------------------------------------------------------------------------------------#


def plot_unit_cell_cross_sections(a, sim_params, custom_material=False):

    resolution, cell_size, geometry = sim_params

    distance_conversion = a / resolution

    if custom_material == True:

        sim = mp.Simulation(
            resolution=resolution, cell_size=cell_size, default_material=geometry
        )

    else:
        sim = mp.Simulation(
            resolution=resolution, cell_size=cell_size, geometry=geometry
        )
    sim.init_sim()
    eps_data = sim.get_epsilon()

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = ax

    z_half_index = int(np.shape(eps_data)[2] // 2)
    y_half_index = int(np.shape(eps_data)[1] // 2)

    dz_index = int(0.2 * np.shape(eps_data)[2])

    fig.suptitle("Cross sections of the 2D Photonic Crystal  Slab")

    extent1 = [
        -y_half_index * distance_conversion,
        y_half_index * distance_conversion,
        -y_half_index * distance_conversion,
        y_half_index * distance_conversion,
    ]

    pos1 = ax1.imshow(
        eps_data[:, :, z_half_index],
        extent=extent1,
        interpolation="spline36",
        cmap="binary",
    )

    extent2 = [
        -y_half_index * distance_conversion,
        y_half_index * distance_conversion,
        -dz_index * distance_conversion,
        dz_index * distance_conversion,
    ]

    pos2 = ax2.imshow(
        np.rot90(
            eps_data[
                :, y_half_index, z_half_index - dz_index : z_half_index + dz_index
            ],
            2,
        ).T,
        extent=extent2,
        interpolation="spline36",
        cmap="binary",
    )
    fig.colorbar(pos2, ax=ax1)

    ax1.set_ylabel("Y distance (nm)")
    ax1.set_xlabel("X distance (nm)")

    ax2.set_ylabel("Z distance (nm)")
    ax2.set_xlabel("X distance (nm)")

    return fig, ax


# ---------------------------------------------------------------------------------------------#
# This function plots a isofrequency diagram for a PCS with square lattice!
# ---------------------------------------------------------------------------------------------#


def isofreq_plot(a, file):

    # Note: a has to be in nanometers!

    f1 = file

    import pandas as pd
    import matplotlib

    fig, ax = plt.subplots(figsize=(14, 10))
    title = "Isofrequency diagram in momentum space"

    light_speed = 3e8

    df = pd.DataFrame(
        dict(x=f1[:-1, 1], y=f1[:-1, 2], z=f1[:-1, -1] * light_speed / (a * 10 ** 3))
    )  # since a is in nm and we want THz
    xcol, ycol, zcol = "x", "y", "z"
    df = df.sort_values(by=[xcol, ycol])
    xvals = df[xcol].unique()
    yvals = df[ycol].unique()
    zvals = df[zcol].values.reshape(len(xvals), len(yvals)).T
    CS = ax.contour(xvals, yvals, zvals, levels=45)
    ax.set_title(title)
    ax.set_xlabel("Normalized $k_x \, a / 2\pi$ ")
    ax.set_ylabel("Normalized $k_y \, a / 2\pi$ ")
    norm = matplotlib.colors.Normalize(vmin=CS.cvalues.min(), vmax=CS.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CS.cmap)
    sm.set_array([])
    fig.colorbar(sm, label="$\\nu$ (THz)")

    plt.text(
        0.015,
        0.02,
        "$\\Gamma$",
        bbox=dict(facecolor="gray", alpha=0.9, boxstyle="Round"),
        fontsize=24,
        color="white",
    )
    plt.text(
        0.015,
        0.47,
        "$Y$",
        bbox=dict(facecolor="gray", alpha=0.9, boxstyle="Round"),
        fontsize=24,
        color="white",
    )
    plt.text(
        0.47,
        0.47,
        "$M$",
        bbox=dict(facecolor="gray", alpha=0.9, boxstyle="Round"),
        fontsize=24,
        color="white",
    )
    plt.text(
        0.47,
        0.02,
        "$X$",
        bbox=dict(facecolor="gray", alpha=0.9, boxstyle="Round"),
        fontsize=24,
        color="white",
    )

    return fig, ax


# ---------------------------------------------------------------------------------------------#
# This function plots a band diagram for a PCS with square lattice!
# ---------------------------------------------------------------------------------------------#


def square_lattice_bands(a, files):

    f2, f3, f4 = files

    fig, ax = plt.subplots(figsize=(14, 10))

    light_speed = 3e8

    w_to_n = light_speed / (a * 10 ** 12)

    for i in range(1, 8):
        if i == 1:
            ax.plot(
                np.linspace(0, len(f2[:, 1]) - 1, len(f2[:, 1])),
                f2[:, -i] * w_to_n,
                c="black",
                linestyle="dashed",
                label="Dispersion modes",
            )
        else:
            ax.plot(
                np.linspace(0, len(f2[:, 1]) - 1, len(f2[:, 1])),
                f2[:, -i] * w_to_n,
                c="black",
                linestyle="dashed",
            )

    ax.plot(
        np.linspace(1, len(f3[:, 1]), len(f3[:, 1])),
        f3[:, -2] * w_to_n,
        c="orange",
        label="TE mode",
    )
    ax.plot(
        np.linspace(1, len(f4[:, 1]), len(f4[:, 1])),
        f4[:, -2] * w_to_n,
        c="blue",
        label="TM mode",
    )

    X_index = int(0.33 * len(f2[:, 1])) + 1
    M_index = int(0.66 * len(f2[:, 1])) + 1

    k_point = np.linspace(0, len(f2[:, 1]) - 1, len(f2[:, 1]))
    Light_line = np.sqrt(f2[:, 1] ** 2 + f2[:, 2] ** 2)

    ax.plot(k_point, np.array(Light_line) * (w_to_n), c="r", label="Light line")
    ax.fill_between(
        k_point,
        Light_line * (w_to_n),
        np.ones_like(k_point) * (w_to_n),
        color="purple",
        alpha=0.9,
        label="Light cone",
    )

    plt.axvline(x=X_index, c="black", alpha=0.3)
    plt.axvline(x=M_index, c="black", alpha=0.3)

    ax.set_xticks([1, X_index, M_index, len(f2[:, 1])])
    ax.set_xticklabels(["$\\Gamma$", "X", "M", "$\\Gamma$"])
    ax.set_xlim([1, len(f2[:, 1])])
    ax.set_ylim([0, 0.75 * (w_to_n)])
    ax.set_xlabel("")
    ax.set_ylabel("Frequency $\\nu$ (THz)")
    ax.legend(loc=2)

    return fig, ax


# ---------------------------------------------------------------------------------------------#
# This function plots a band diagram for a PCS with hexagonal lattice!
# ---------------------------------------------------------------------------------------------#


def hexagonal_lattice_bands(a, files):

    f1, f2, f3 = files

    fig, ax = plt.subplots(figsize=(14, 10))

    light_speed = 3e8

    w_to_n = light_speed / (a * 10 ** 12)

    for i in range(1, 7):

        if i == 1:
            ax.plot(
                np.linspace(0, len(f1[:, 1]) - 1, len(f1[:, 1])),
                f1[:, -i] * (w_to_n),
                c="black",
                linestyle="dashed",
                label="Dispersion modes",
            )
        else:
            ax.plot(
                np.linspace(0, len(f1[:, 1]) - 1, len(f1[:, 1])),
                f1[:, -i] * (w_to_n),
                c="black",
                linestyle="dashed",
            )

    ax.plot(
        np.linspace(0, len(f2[:, 1]) - 1, len(f2[:, 1])),
        f2[:, -1] * (w_to_n),
        c="orange",
        label="TE mode",
    )
    ax.plot(
        np.linspace(0, len(f2[:, 1]) - 1, len(f2[:, 1])),
        f2[:, -2] * (w_to_n),
        c="orange",
    )
    ax.plot(
        np.linspace(0, len(f3[:, 1]) - 1, len(f3[:, 1])),
        f3[:, -1] * (w_to_n),
        c="blue",
        label="TM mode",
    )
    ax.plot(
        np.linspace(0, len(f3[:, 1]) - 1, len(f3[:, 1])), f3[:, -2] * (w_to_n), c="blue"
    )

    K_index = int(0.33 * len(f1[:, 1]))
    M_index = int(0.66 * len(f1[:, 1]))
    k_point = np.linspace(0, len(f1[:, 1]) - 1, len(f1[:, 1]))

    k_till_K = np.array(
        [
            np.linspace(0, 1 / np.sqrt(3), K_index + 1),
            np.linspace(0, 1 / 3, K_index + 1),
        ]
    )
    k_till_M = np.array(
        [
            1 / np.sqrt(3) * np.ones(M_index - K_index),
            np.linspace(1 / 3, 0, M_index - K_index),
        ]
    )
    k_till_Gamma = np.array(
        [
            np.linspace(1 / np.sqrt(3), 0, len(f1[:, 1]) - M_index),
            np.zeros(len(f1[:, 1]) - M_index),
        ]
    )

    ax.fill_between(
        k_point,
        np.max(f2[:, -2]) * np.ones_like(k_point) * (w_to_n),
        np.min(f2[K_index:M_index, -1]) * np.ones_like(k_point) * (w_to_n),
        color="orange",
        alpha=0.5,
        label="TE Gap",
    )
    ks = np.hstack([k_till_K, k_till_M[:, 1:], k_till_Gamma])
    Light_line = np.sqrt(ks[0, :] ** 2 + ks[1, :] ** 2)

    ax.plot(k_point, np.array(Light_line) * (w_to_n), c="r", label="Light line")
    ax.fill_between(
        k_point,
        Light_line * (w_to_n),
        np.ones_like(k_point) * (w_to_n),
        color="purple",
        alpha=0.9,
        label="Light cone",
    )

    plt.axvline(x=K_index, c="black", alpha=0.3)
    plt.axvline(x=M_index, c="black", alpha=0.3)

    ax.set_xticks([1, K_index, M_index, len(f1[:, 1])])
    ax.set_xticklabels(["$\\Gamma$", "K", "M", "$\\Gamma$"])
    ax.set_xlim([0, len(f1[:, 1]) - 1])
    ax.set_ylim([0, 0.75 * (w_to_n)])
    ax.set_xlabel("")
    ax.set_ylabel("Frequency $\\nu$ (THz)")
    ax.legend(loc=2)

    return fig, ax


def hexagonal_lattice_bands_1(a, files):

    f1, f2, f3 = files

    fig, ax = plt.subplots(figsize=(14, 10))

    light_speed = 3e8

    w_to_n = light_speed / (a * 10 ** 12)

    for i in range(1, 7):

        if i == 1:
            ax.plot(
                np.linspace(0, len(f1[:, 1]) - 1, len(f1[:, 1])),
                f1[:, -i] * (w_to_n),
                c="black",
                linestyle="dashed",
                label="Dispersion modes",
            )
        else:
            ax.plot(
                np.linspace(0, len(f1[:, 1]) - 1, len(f1[:, 1])),
                f1[:, -i] * (w_to_n),
                c="black",
                linestyle="dashed",
            )

    ax.plot(
        np.linspace(0, len(f1[:, 1]) - 1, len(f1[:, 1])),
        f1[:, -6] * (w_to_n),
        c="orange",
        label="TE mode",
    )
    ax.plot(
        np.linspace(0, len(f1[:, 1]) - 1, len(f1[:, 1])),
        f1[:, -4] * (w_to_n),
        c="orange",
    )
    ax.plot(
        np.linspace(0, len(f1[:, 1]) - 1, len(f1[:, 1])),
        f1[:, -5] * (w_to_n),
        c="blue",
        label="TM mode",
    )
    ax.plot(
        np.linspace(0, len(f1[:, 1]) - 1, len(f1[:, 1])), f1[:, -3] * (w_to_n), c="blue"
    )

    K_index = int(0.33 * len(f1[:, 1]))
    M_index = int(0.66 * len(f1[:, 1]))
    k_point = np.linspace(0, len(f1[:, 1]) - 1, len(f1[:, 1]))

    k_till_K = np.array(
        [
            np.linspace(0, 1 / np.sqrt(3), K_index + 1),
            np.linspace(0, 1 / 3, K_index + 1),
        ]
    )
    k_till_M = np.array(
        [
            1 / np.sqrt(3) * np.ones(M_index - K_index),
            np.linspace(1 / 3, 0, M_index - K_index),
        ]
    )
    k_till_Gamma = np.array(
        [
            np.linspace(1 / np.sqrt(3), 0, len(f1[:, 1]) - M_index),
            np.zeros(len(f1[:, 1]) - M_index),
        ]
    )

    # ax.fill_between(k_point, np.max(f1[:,-6])*np.ones_like(k_point)*(w_to_n), np.min(f1[K_index:M_index,-4])*np.ones_like(k_point)*(w_to_n), color='orange', alpha=0.5, label='TE Gap')
    ks = np.hstack([k_till_K, k_till_M[:, 1:], k_till_Gamma])
    Light_line = np.sqrt(ks[0, :] ** 2 + ks[1, :] ** 2)

    ax.plot(k_point, np.array(Light_line) * (w_to_n), c="r", label="Light line")
    ax.fill_between(
        k_point,
        Light_line * (w_to_n),
        np.ones_like(k_point) * (w_to_n),
        color="purple",
        alpha=0.9,
        label="Light cone",
    )

    plt.axvline(x=K_index, c="black", alpha=0.3)
    plt.axvline(x=M_index, c="black", alpha=0.3)

    ax.set_xticks([1, K_index, M_index, len(f1[:, 1])])
    ax.set_xticklabels(["$\\Gamma$", "K", "M", "$\\Gamma$"])
    ax.set_xlim([0, len(f1[:, 1]) - 1])
    ax.set_ylim([0, 0.75 * (w_to_n)])
    ax.set_xlabel("")
    ax.set_ylabel("Frequency $\\nu$ (THz)")
    ax.legend(loc=2)

    return fig, ax


# ---------------------------------------------------------------------------------------------#
# This function the intensity across a  line in a PCS
# ---------------------------------------------------------------------------------------------#


def plot_intentsity_in_z_axis(field, a, resolution, num_band, zlim, title):

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title(title)

    distance_conversion = resolution / a

    zmin, zmax = zlim

    z_offset = int(np.array(np.shape(field))[3] // 2)

    zmin_index = int(zmin * distance_conversion) + z_offset
    zmax_index = int(zmax * distance_conversion) + z_offset

    z = np.linspace(zmin, zmax, zmax_index - zmin_index)

    x_half_index = int(np.shape(field)[1] // 2)
    y_half_index = int(np.shape(field)[2] // 2)

    intens = field[num_band][x_half_index, y_half_index, zmin_index:zmax_index]

    ax.scatter(z, intens)
    ax.set_ylabel("I")
    ax.set_xlabel("z (nm)")

    return fig, ax
