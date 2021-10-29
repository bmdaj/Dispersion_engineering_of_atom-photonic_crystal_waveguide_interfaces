import meep as mp
from meep import mpb

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

resolution = 32

Med_1 = mp.Medium(index=np.sqrt(12))
Med_2 = mp.Medium(index=1)

a = 1  #Arbitrary units

width = 0.4 * a

sc = 1 * a  

y_lim = 15

geometry_lattice = mp.Lattice(size=mp.Vector3(sc,y_lim))
geometry = [mp.Block(size=mp.Vector3(width, width),
            center=mp.Vector3(0.5*(width-sc),0), material=Med_1)]


cell_size = mp.Vector3(sc,y_lim)

num_bands = 10

num_k = 20
k_min = 0
k_max = 1

k_points = mp.interpolate(num_k, [mp.Vector3(k_min), mp.Vector3(k_max)])


ms = mpb.ModeSolver(geometry_lattice=geometry_lattice,
                    geometry=geometry,
                    k_points=k_points,
                    resolution=resolution,
                    num_bands=num_bands);
ms.run();

ms = mpb.ModeSolver(geometry_lattice=geometry_lattice,
                    geometry=geometry,
                    k_points=k_points,
                    resolution=resolution,
                    num_bands=num_bands);
ms.run();