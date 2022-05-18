import meep as mp
from meep import mpb
import numpy as np
import matplotlib.pyplot as plt

def geom_hexagon_glass_sub_PCS (params, cell):
    
    # Extraction of parameters. See PNAS paper.
    
    a, t, scaling = params
    
    sx = np.array(cell)[0]
    sy = np.array(cell)[1]
    sz = np.array(cell)[2]
    
    glass_t = 500 / a # 500 nm subtrate
    
    geometry = [mp.Block(center=mp.Vector3(0,0,0), size=mp.Vector3(mp.inf,mp.inf,t), material=mp.Medium(index = 1.9935))]
    geometry.append(mp.Block(center=mp.Vector3(0,0,-t/2-glass_t/2), size=mp.Vector3(mp.inf,mp.inf,glass_t), material=mp.Medium(index = 1.45)))

    # A hexagonal prism defined by six vertices centered on the origin
    # of material crystalline silicon (from the materials library)


    vertices = [mp.Vector3(scaling*2/np.sqrt(3),0),
                mp.Vector3(scaling*1/np.sqrt(3),scaling*1),
                mp.Vector3(-scaling*1/np.sqrt(3),scaling*1),
                mp.Vector3(scaling*-2/np.sqrt(3),0),
                mp.Vector3(-scaling*1/np.sqrt(3),-scaling*1),
                mp.Vector3(scaling*1/np.sqrt(3),-scaling*1)]

    vertices_new = []
    
    # we change the vertices to match the basis of our lattice

    geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1, sz),
                              basis1=mp.Vector3(np.sqrt(3) / 2, 0.5, 0.0),
                              basis2=mp.Vector3(np.sqrt(3) / 2, -0.5, 0.0))
    vertices_new = []
    for vertice in vertices:
        vertices_new.append(mp.cartesian_to_lattice(vertice, geometry_lattice))

    geometry.append(mp.Prism(vertices_new, height=t, center=mp.Vector3(), material=mp.Medium(index = 1)))

    return geometry, geometry_lattice

resolution = 64
num_bands = 6

a = 405 # nm
t = 180 / a
scaling = (1-t)*0.5

sx = 1 
sy = 1 
sz = 10
cell = mp.Vector3(sx, sy, sz)

parameters = [a, t, scaling]
geometry, geometry_lattice = geom_hexagon_glass_sub_PCS (parameters, cell)

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