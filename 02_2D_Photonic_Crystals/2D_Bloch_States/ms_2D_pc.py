import meep as mp
from meep import mpb
import numpy as np

resolution = 32

Med_1 = mp.Medium(index=np.sqrt(8.9)) #as in alumina
Med_2 = mp.Medium(index=1)

a = 1 # arbitrary units
sc = 1 * a  

geometry_lattice = mp.Lattice(size=mp.Vector3(sc, sc))

r  = 0.2 * a
geometry = [mp.Cylinder(r, material=Med_1)]

num_bands = 12


k_points = [mp.Vector3(),          # Gamma
            mp.Vector3(0.5),       # X
            mp.Vector3(0.5, 0.5),  # M
            mp.Vector3()]          # Gamma

k_points = mp.interpolate(40, k_points)

ms = mpb.ModeSolver(geometry_lattice=geometry_lattice,
                    geometry=geometry,
                    k_points=k_points,
                    resolution=resolution,
                    num_bands=num_bands);
ms.run();