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

resolution = 128 

a = 1 # arbitrary units
w1 = 0.2 * a
w2 = 0.8 * a # For the Quarter Wave Stack: w2 = a - w1

Med_1 = mp.Medium(index=np.sqrt(13))
Med_2= mp.Medium(index=1)

sc_z = 1 * a  # supercell width

geometry_lattice = mp.Lattice(size=mp.Vector3(0,0,sc_z))

center1 = 0.5*(-sc_z + w1)
center2 = 0.5*(sc_z - w2)
geometry = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, w1),
                     center=mp.Vector3(0,0,center1), material=Med_1),mp.Block(size=mp.Vector3(mp.inf, mp.inf, w2),
                     center=mp.Vector3(0,0,center2), material=Med_2)]

num_bands = 10

num_k = 40
k_z = 0.5 #This is the one that has to be changed
k_miny = 0.0
k_maxy = 1.5

k_points = mp.interpolate(num_k, [mp.Vector3(0,k_miny,k_z), mp.Vector3(0,k_maxy,k_z)])

ms = mpb.ModeSolver(geometry_lattice=geometry_lattice,
                    geometry=geometry,
                    k_points=k_points,
                    resolution=resolution,
                    num_bands=num_bands)
ms.run()