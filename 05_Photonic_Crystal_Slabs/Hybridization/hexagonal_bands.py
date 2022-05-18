import meep as mp
from meep import mpb

import numpy as np

resolution = 64
num_bands = 6

a = 405 # nm
t = 180 / a
g = 250 / a

sx = 1 
sy = 1 
sz = 10
cell = mp.Vector3(sx, sy, sz)

geometry = [mp.Block(center=mp.Vector3(0,0,0.5*g), size=mp.Vector3(mp.inf,mp.inf,t), material=mp.Medium(index = 1.9935))]
geometry.append(mp.Block(center=mp.Vector3(0,0,-0.5*g), size=mp.Vector3(mp.inf,mp.inf,t), material=mp.Medium(index = 1.9935)))

# A hexagonal prism defined by six vertices centered on the origin
# of material crystalline silicon (from the materials library)

multiplier = (1-t)*0.5

vertices = [mp.Vector3(multiplier*2/np.sqrt(3),0),
            mp.Vector3(multiplier*1/np.sqrt(3),multiplier*1),
            mp.Vector3(-multiplier*1/np.sqrt(3),multiplier*1),
            mp.Vector3(multiplier*-2/np.sqrt(3),0),
            mp.Vector3(-multiplier*1/np.sqrt(3),-multiplier*1),
            mp.Vector3(multiplier*1/np.sqrt(3),-multiplier*1)]

vertices_new = []

geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1, sz),
                              basis1=mp.Vector3(np.sqrt(3) / 2, 0.5, 0.0),
                              basis2=mp.Vector3(np.sqrt(3) / 2, -0.5, 0.0))
vertices_new = []
for vertice in vertices:
    vertices_new.append(mp.cartesian_to_lattice(vertice, geometry_lattice))

geometry.append(mp.Prism(vertices_new, height=t, center=mp.Vector3(0,0,0.5*g), material=mp.Medium(index = 1)))
geometry.append(mp.Prism(vertices_new, height=t, center=mp.Vector3(0,0,-0.5*g), material=mp.Medium(index = 1)))



k_points = [mp.Vector3(),                      # Gamma
            mp.Vector3(2/3, 1/3),              # K
            mp.Vector3(0.5, 0.5),              # M
            mp.Vector3()]                      # Gamma

num_k = 30

k_points = mp.interpolate(num_k, k_points)

ms = mpb.ModeSolver(geometry_lattice=geometry_lattice,
                    geometry=geometry,
                    k_points=k_points,
                    resolution=resolution,
                    num_bands=num_bands);
ms.run();