import meep as mp
from meep import mpb
import numpy as np
import matplotlib.pyplot as plt

def geom_hexagon_PCS_glass_sub_slope (params):
    
    # Extraction of parameters
    
    dielec_const, a, t, alpha, geometry_lattice = params
    
    def  geometry(Vector3):
        
        array3 = np.array(mp.lattice_to_cartesian(Vector3, geometry_lattice))
        y = np.abs(array3 [0])
        x = np.abs(array3 [1])
        z = array3 [2]
    
    
        r1 = (1 - t)/ 2 #  upper radius (biggest one)
        r2 = r1 - t * np.sin(alpha)
            
        r = np.abs(0.5*(r1+r2)+ z*(r1-r2)/(t)) #this defines the slope of the walls
        h = r / np.sqrt(3) # r tan(30º)

    
        condition_x_square = np.abs(x) <= r
        condition_y_square = np.abs(y) <= h
    
        condition_x_triangle = np.abs(x) <= r
        condition_y_triangle = h <= np.abs(y) <= -h/r * np.abs(x) + 2*h
            
        condition_z = np.abs(z) <= t*0.5
    
    
        if condition_x_square and condition_y_square and condition_z:
        
            eps = mp.Medium(index = 1) 

        
        elif condition_x_triangle and condition_y_triangle and condition_z:
        
            eps = mp.Medium(index = 1)  
            
        elif  condition_z:
            
            eps = mp.Medium(index = dielec_const)

        else:
        
            eps = mp.Medium(index = 1) 
        
        if -500/a - t/2 < z < -t /2:
         
            eps = mp.Medium(index = 1.45) 
        
        return eps

    return geometry


resolution = 32
num_bands = 5

a = 405 # nm
t = 180 / a
scaling = (1-t)*0.5

sx = 1 
sy = 1 
sz = 10
cell = mp.Vector3(sx, sy, sz)

custom_material = True

geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1, sz),
                              basis1=mp.Vector3(np.sqrt(3) / 2, 0.5, 0.0),
                              basis2=mp.Vector3(np.sqrt(3) / 2, -0.5, 0.0))

dielec_const = 1.9935
alpha = 7.9 * np.pi / 180
params = dielec_const, a, t, alpha, geometry_lattice

eps_func = geom_hexagon_PCS_glass_sub_slope(params)
default_material = eps_func

k_points = [mp.Vector3(),                      # Gamma
            mp.Vector3(2/3, 1/3),              # K
            mp.Vector3(0.5, 0.5),              # M
            mp.Vector3()]                      # Gamma

num_k = 30

k_points = mp.interpolate(num_k, k_points)

ms = mpb.ModeSolver(geometry_lattice=geometry_lattice,
                    default_material=default_material,
                    k_points=k_points,
                    resolution=resolution,
                    num_bands=num_bands);
ms.run();