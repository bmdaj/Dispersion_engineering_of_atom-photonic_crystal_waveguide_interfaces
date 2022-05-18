# %%
import meep as mp
from meep import mpb
import numpy as np

# %%
def freq_to_meep(freq, a):
    "Calculate frequency in MEEP units"
    return freq * a / 3e8


def first_te_gap(params):

    multiplier, a = params

    freq = 351.726e12

    transition_freq = freq_to_meep(freq, a * 1e-9)

    t = 200 / a

    ms.mesh_size = 7

    geometry = [
        mp.Block(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(mp.inf, mp.inf, t),
            material=mp.Medium(index=1.9935),
        )
    ]

    # A hexagonal prism defined by six vertices centered on the origin
    # of material crystalline silicon (from the materials library)

    # multiplier = (1 - t) * 0.5

    vertices = [
        mp.Vector3(multiplier * 2 / np.sqrt(3), 0),
        mp.Vector3(multiplier * 1 / np.sqrt(3), multiplier * 1),
        mp.Vector3(-multiplier * 1 / np.sqrt(3), multiplier * 1),
        mp.Vector3(multiplier * -2 / np.sqrt(3), 0),
        mp.Vector3(-multiplier * 1 / np.sqrt(3), -multiplier * 1),
        mp.Vector3(multiplier * 1 / np.sqrt(3), -multiplier * 1),
    ]

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
            vertices_new,
            height=float(t),
            center=mp.Vector3(),
            material=mp.Medium(index=1),
        )
    )

    ms.geometry = geometry

    M_point = (mp.Vector3(0.5, 0.5),)  # M
    K_point = (mp.Vector3(2 / 3, 1 / 3),)  # K

    k_points = M_point
    ms.k_points = k_points

    ms.run_te()

    band_2_min = ms.freqs[1]

    ms.k_points = K_point
    ms.run_te()

    band_1_max = ms.freqs[0]

    band_gap = band_2_min - band_1_max

    # Now we want to allign the midgap frequency with the transition frequency
    if band_gap > 0:

        midgap_freq = 0.5 * (band_2_min + band_1_max)
        cost_function = np.abs(midgap_freq - transition_freq)
    else:
        cost_function = 10

    # cost_function = -band_gap

    print("Cost function: ", cost_function)

    return cost_function


def calculate_freqs(params):

    multiplier, a = params

    t = 200 / a

    ms.mesh_size = 7

    geometry = [
        mp.Block(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(mp.inf, mp.inf, t),
            material=mp.Medium(index=1.9935),
        )
    ]

    # A hexagonal prism defined by six vertices centered on the origin
    # of material crystalline silicon (from the materials library)

    # multiplier = (1 - t) * 0.5

    vertices = [
        mp.Vector3(multiplier * 2 / np.sqrt(3), 0),
        mp.Vector3(multiplier * 1 / np.sqrt(3), multiplier * 1),
        mp.Vector3(-multiplier * 1 / np.sqrt(3), multiplier * 1),
        mp.Vector3(multiplier * -2 / np.sqrt(3), 0),
        mp.Vector3(-multiplier * 1 / np.sqrt(3), -multiplier * 1),
        mp.Vector3(multiplier * 1 / np.sqrt(3), -multiplier * 1),
    ]

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
            vertices_new,
            height=float(t),
            center=mp.Vector3(),
            material=mp.Medium(index=1),
        )
    )

    ms.geometry = geometry

    M_point = (mp.Vector3(0.5, 0.5),)  # M
    K_point = (mp.Vector3(2 / 3, 1 / 3),)  # K

    k_points = M_point
    ms.k_points = k_points

    ms.run_te()

    band_2_min = ms.freqs[1]

    ms.k_points = K_point
    ms.run_te()

    band_1_max = ms.freqs[0]

    tm_band_freq = band_2_min

    midgap_freq = 0.5 * (band_2_min + band_1_max)

    band_gap_width = band_2_min - band_1_max

    return tm_band_freq, midgap_freq, band_gap_width


# %%

# We start with the case of Sr

freq = 351.726e12  # Cs line

a = 400  # nm, EXAMPLE

freq_meep = freq_to_meep(freq, a * 1e-9)

print("Frequency in MEEP units: ", freq_meep)

# %%

resolution = 64
num_bands = 2

t = 200 / a

sx = 1
sy = 1
sz = 10
cell = mp.Vector3(sx, sy, sz)

geometry = [
    mp.Block(
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(mp.inf, mp.inf, t),
        material=mp.Medium(index=1.9935),
    )
]

# A hexagonal prism defined by six vertices centered on the origin
# of material crystalline silicon (from the materials library)

multiplier = (1 - t) * 0.5

vertices = [
    mp.Vector3(multiplier * 2 / np.sqrt(3), 0),
    mp.Vector3(multiplier * 1 / np.sqrt(3), multiplier * 1),
    mp.Vector3(-multiplier * 1 / np.sqrt(3), multiplier * 1),
    mp.Vector3(multiplier * -2 / np.sqrt(3), 0),
    mp.Vector3(-multiplier * 1 / np.sqrt(3), -multiplier * 1),
    mp.Vector3(multiplier * 1 / np.sqrt(3), -multiplier * 1),
]

vertices_new = []

geometry_lattice = mp.Lattice(
    size=mp.Vector3(1, 1, sz),
    basis1=mp.Vector3(np.sqrt(3) / 2, 0.5),
    basis2=mp.Vector3(
        np.sqrt(3) / 2,
        -0.5,
    ),
)
vertices_new = []
for vertice in vertices:
    vertices_new.append(mp.cartesian_to_lattice(vertice, geometry_lattice))

geometry.append(
    mp.Prism(vertices_new, height=t, center=mp.Vector3(), material=mp.Medium(index=1))
)


M_point = (mp.Vector3(0.5, 0.5),)  # M
K_point = (mp.Vector3(2 / 3, 1 / 3),)  # K


k_points = M_point

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    k_points=k_points,
    resolution=resolution,
    num_bands=num_bands,
)

ms.run_te()

# %%

band_2_min = ms.freqs[1]

# %%

ms.k_points = K_point
ms.run_te()

# %%
band_1_max = ms.freqs[0]

band_gap = band_2_min - band_1_max

print("gap size: {}".format(band_gap))

# %%

from scipy.optimize import minimize

x0 = [multiplier, a]

cons = (
    {
        "type": "ineq",
        "fun": lambda x: x[0] * x[1] - 50,
    },  # diameter holes no smaller than 50 nm
)

result = minimize(
    first_te_gap,
    x0,
    method="Nelder-Mead",
    bounds=[[0.1, 0.8], [100, 1000]],
    options={"xatol": 1e-6, "fatol": 1e-6},
    constraints=cons,
)

# %%

print("Hexagon size at maximum: {}".format(result.x[0]))
print("Lattice constant at maximum: {}".format(result.x[1]))
print("Cost function at maximum: {}".format(result.fun))

# %%

params = [result.x[0], result.x[1]]

tm_band_freq, midgap_freq, band_gap_width = calculate_freqs(params)
print("TM band freq.: {} meep units".format(tm_band_freq))
print("Midgap freq : {} meep units".format(midgap_freq))
print("Band gap width freq : {} meep units".format(band_gap_width))
