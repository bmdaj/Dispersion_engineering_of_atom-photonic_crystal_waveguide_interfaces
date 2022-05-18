import meep as mp
from meep import mpb

import numpy as np


# Some parameters to describe the geometry:

a = 335e-9  # in nm
w = 335e-9 / a
t = 200e-9 / a
r = 0.5 * 116e-9 / a
g = 250e-9 / a
index = 1.9935  # refractive index of Si3N4
az = 15

# The cell dimensions
sx = a / a
sy = 15     #size of cell in y direction (perpendicular to wvg.)
sz = 15     # size of cell in z direction (perpendicular to wvg.)


cell = mp.Vector3(sx, sy, sz)
geometry_lattice = mp.Lattice(size=mp.Vector3(sx,sy, sz))


resolution = 16

num_bands = 4

num_k = 25
k_min = 0.3
k_max = 0.5 

k_points = mp.interpolate(num_k, [mp.Vector3(k_min), mp.Vector3(k_max)])

geometry = [mp.Block(center=mp.Vector3(0,-0.5*(g+w),0), 
            size=mp.Vector3(mp.inf,w,t), 
            material=mp.Medium(index = 2))]                    # First Si3N4 waveguide block

geometry.append(mp.Block(center=mp.Vector3(0,0.5*(g+w),0), 
            size=mp.Vector3(mp.inf,w,t), 
            material=mp.Medium(index = 2)))  

for i in range(int(sx)):
    geometry.append(mp.Cylinder(center=mp.Vector3(int(0.5*i),0.5*(g+w),0),
                               height = t,
                               radius = r,
                               material = mp.Medium(index = 1)))
    geometry.append(mp.Cylinder(center=mp.Vector3(int(0.5*i),-0.5*(g+w),0),
                               height = t,
                               radius = r,
                               material = mp.Medium(index = 1)))
    geometry.append(mp.Cylinder(center=mp.Vector3(-int(0.5*i),-0.5*(g+w),0),
                               height = t,
                               radius = r,
                               material = mp.Medium(index = 1)))
    geometry.append(mp.Cylinder(center=mp.Vector3(-int(0.5*i),0.5*(g+w),0),
                               height = t,
                               radius = r,
                               material = mp.Medium(index = 1)))


ms = mpb.ModeSolver(geometry_lattice=geometry_lattice,
                    k_points=k_points,
                    geometry=geometry,
                    resolution=resolution,
                    num_bands=num_bands);
ms.run_zeven();