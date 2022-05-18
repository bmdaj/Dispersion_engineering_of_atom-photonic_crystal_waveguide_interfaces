import meep as mp
from meep import mpb
import numpy as np

resolution = 512

a = 1 #arbitrary units
w = 0.5 * a

Med_1 = mp.Medium(index=np.sqrt(13))
Med_2_array = np.array([mp.Medium(index=np.sqrt(13)),mp.Medium(index=np.sqrt(12)),mp.Medium(index=1)])
num = len(Med_2_array)

sc_z = 1 * a  # supercell width

geometry_lattice = mp.Lattice(size=mp.Vector3(0,0,sc_z))

geometry_array = list()
for i in range(num):
    geometry_array.append([mp.Block(size=mp.Vector3(mp.inf, mp.inf, w),
                     center=mp.Vector3(0,0,-0.5*w), material=Med_1),mp.Block(size=mp.Vector3(mp.inf, mp.inf, w),
                     center=mp.Vector3(0,0,+0.5*w), material=Med_2_array[i])])
num_bands = 8

num_k = 400
k_min = -0.5
k_max = 0.5
k_points = mp.interpolate(num_k, [mp.Vector3(0,0,k_min), mp.Vector3(0,0,k_max)])

ms = mpb.ModeSolver(geometry_lattice=geometry_lattice,
                    geometry=geometry_array[2],
                    k_points=k_points,
                    resolution=resolution,
                    num_bands=num_bands)
ms.run()