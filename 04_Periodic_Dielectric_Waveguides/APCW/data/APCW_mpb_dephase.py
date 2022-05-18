import meep as mp
from meep import mpb
import numpy as np

# Some parameters to describe the geometry:
index = 1.9935  # refractive index of Si3N4
ax = 3.7   #  e-7 meters
ay = 15
az = 15

# The cell dimensions
sx = 3.7 / ax
sy = 15     #size of cell in y direction (perpendicular to wvg.)
sz = 15     # size of cell in z direction (perpendicular to wvg.)


cell = mp.Vector3(sx, sy, sz)
geometry_lattice = mp.Lattice(size=mp.Vector3(sx,sy, sz))


resolution = 12

def eps_func (Vector3):
    array3 = np.array(Vector3)
    x = array3 [0]
    y = array3 [1]
    z = array3 [2]
    

    dielec_const = 1.9935 # this can be modified
    
    dephase = 180.0 * np.pi / 180
    period = 3.7 / ax
    amp = 1.4 /ax
    width = 2.88 /ax
    thickness = 2 /ax
    gap = 2.2 /ax
    
    if y >= 0.5*gap and amp*np.cos(2*np.pi*(x) + dephase) + 0.5*gap+width>= y and 0.5 *thickness > np.abs(z):
        eps = mp.Medium(index = dielec_const) 
        
    elif y <= -0.5*width and -amp*np.cos(2*np.pi*(x)) - 0.5*gap-width <= y and 0.5 *thickness > np.abs(z):
        eps = mp.Medium(index = dielec_const) 
    else:
        eps = mp.Medium(epsilon = 1)
        
    if  width-amp+0.5*gap  >= np.abs(y) >= 0.5*gap and 0.5 *thickness > np.abs(z) :
        eps = mp.Medium(index = dielec_const) 

    return eps

default_material = eps_func

num_bands = 10

num_k = 40
k_min = 0.435
k_max = 0.5 

k_points = mp.interpolate(num_k, [mp.Vector3(k_min), mp.Vector3(k_max)])

ms = mpb.ModeSolver(geometry_lattice=geometry_lattice,
                    k_points=k_points,
                    default_material=default_material,
                    resolution=resolution,
                    num_bands=num_bands);

ms.run(); 