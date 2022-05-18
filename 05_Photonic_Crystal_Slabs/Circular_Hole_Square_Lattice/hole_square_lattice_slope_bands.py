import numpy as np
import meep as mp
from meep import mpb

def eps_func (Vector3):
    array3 = np.array(Vector3)
    x = array3 [0]
    y = array3 [1]
    z = array3 [2]
    

    dielec_const = 1.9935 # this can be modified
    
    h = t #  nanometers
    alpha = 7.9 * np.pi / 180 # degrees to radians
    r1 = r #  upper radius (biggest one)
    r2 = r1 - h * np.sin(alpha)
    
        
    x_prime = 0.5*(r1+r2)+ z*(r1-r2)/h #this defines the slope of the walls
    y_prime = x_prime                  #the slope is the same in both directions (simmetry)
    
    if x**2 + y**2 > x_prime**2 and np.abs(z) < h*0.5:
        
        eps = mp.Medium(index = dielec_const) 
        
    else:
        
        eps = mp.Medium(index = 1) 
        
    return eps

resolution = 24
num_bands = 8

a = 290 # nm
r = 103 / a 
t = 200 / a

sx = 1 
sy = 1 
sz = 10 
cell = mp.Vector3(sx, sy, sz)
geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1, sz))

default_material = eps_func


num_k = 20

k_points = [mp.Vector3(),          # Gamma
            mp.Vector3(0.5),       # X
            mp.Vector3(0.5, 0.5),  # M
            mp.Vector3()]          # Gamma

k_points = mp.interpolate(num_k, k_points)
        
ms = mpb.ModeSolver(geometry_lattice=geometry_lattice,
                k_points=k_points,
                default_material=default_material,
                resolution=resolution,
                num_bands=num_bands);
ms.run();