import meep as mp
from meep import mpb
import numpy as np

import matplotlib.pyplot as plt

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
k_minz = -0.5
k_maxz = 0.0
k_miny = 0.0
k_maxy = 1.5

k_pointsz = mp.interpolate(num_k, [mp.Vector3(0,0,k_minz), mp.Vector3(0,0,k_maxz)])
k_pointsy = mp.interpolate(num_k, [mp.Vector3(0,k_miny,k_maxz), mp.Vector3(0,k_maxy,k_maxz)])
k_points = k_pointsz + k_pointsy

ms = mpb.ModeSolver(geometry_lattice=geometry_lattice,
                    geometry=geometry,
                    k_points=k_points,
                    resolution=resolution,
                    num_bands=num_bands)
ms.run()