import meep as mp
from meep import mpb

import numpy as np

resolution = 32

num_bands = 1

a = 290 # nm
r = 103 / a 
t = 200 / a

sx = 1 
sy = 1 
sz = 10 
cell = mp.Vector3(sx, sy, sz)
geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1, sz))

geometry = [mp.Block(center=mp.Vector3(0,0,0), size=mp.Vector3(mp.inf,mp.inf,t), material=mp.Medium(index = 1.9935))]
geometry.append(mp.Cylinder(radius=r, height = t, material=mp.Medium(epsilon=1)))

num_k = 40

k_points = []

for i in range(num_k+1):
    for j in range(num_k+1):
        k_points.append(mp.Vector3(0.5*(i/num_k),0.5*(j/num_k)))
        
ms = mpb.ModeSolver(geometry_lattice=geometry_lattice,
                    geometry=geometry,
                    k_points=k_points,
                    resolution=resolution,
                    num_bands=num_bands);
ms.run();