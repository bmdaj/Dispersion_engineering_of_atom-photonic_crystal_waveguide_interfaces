# %%
# --------------------------------------------------------------------------------#
# IMPORTS
# --------------------------------------------------------------------------------#

import meep as mp 
import numpy as np
import matplotlib.pyplot as plt
from hexalattice.hexalattice import *
import scipy.optimize as opt

plt.rcParams.update({'font.size': 16})

# --------------------------------------------------------------------------------#
# FUNCTIONS
# --------------------------------------------------------------------------------#


def create_structure(s1,s2,dR):

    a = 405e-9 # nm
    t = 180e-9 / a

    length = 30
    N = 27

    dpml = 1

    geometry = [mp.Block(center=mp.Vector3(0,0,0), 
            size=mp.Vector3(length-2*dpml,length-2*dpml,t), 
            material=mp.Medium(index = 2))]     

    
    multiplier = (1-t)*0.5
    multiplier = multiplier*1/np.sqrt(3)


    vertices = [mp.Vector3(0,multiplier*2/np.sqrt(3)),
                mp.Vector3(multiplier*1,multiplier*1/np.sqrt(3)),
                mp.Vector3(multiplier*1,-multiplier*1/np.sqrt(3)),
                mp.Vector3(0,multiplier*-2/np.sqrt(3)),
                mp.Vector3(-multiplier*1,-multiplier*1/np.sqrt(3)),
                mp.Vector3(-multiplier*1,multiplier*1/np.sqrt(3))]

    multiplier = (1-t)*0.5
    multiplier_NN = multiplier*1/np.sqrt(3) * s2


    vertices_NN = [mp.Vector3(0,multiplier_NN*2/np.sqrt(3)),
                mp.Vector3(multiplier_NN*1,multiplier_NN*1/np.sqrt(3)),
                mp.Vector3(multiplier_NN*1,-multiplier_NN*1/np.sqrt(3)),
                mp.Vector3(0,multiplier_NN*-2/np.sqrt(3)),
                mp.Vector3(-multiplier_NN*1,-multiplier_NN*1/np.sqrt(3)),
                mp.Vector3(-multiplier_NN*1,multiplier_NN*1/np.sqrt(3))]
 
    multiplier_0 = multiplier*1/np.sqrt(3) * s1


    vertices_0 = [mp.Vector3(0,multiplier_0*2/np.sqrt(3)),
                mp.Vector3(multiplier_0*1,multiplier_0*1/np.sqrt(3)),
                mp.Vector3(multiplier_0*1,-multiplier_0*1/np.sqrt(3)),
                mp.Vector3(0,multiplier_0*-2/np.sqrt(3)),
                mp.Vector3(-multiplier_0*1,-multiplier_0*1/np.sqrt(3)),
                mp.Vector3(-multiplier_0*1,multiplier_0*1/np.sqrt(3))]
    


    hex_centers, _ = create_hex_grid(nx=N, ny=N, crop_circ=N//2)

    for center in hex_centers: 
        center_final = mp.Vector3(center[0], center[1])

        if np.sqrt(center[0]**2+center[1]**2) <= 0.1:
            
            geometry.append(mp.Prism(vertices_0,
                                height=t,
                                center=center_final,
                                material=mp.Medium(index = 1))) #add the hexagons in the lattice  

            continue

        if np.sqrt(center[0]**2+center[1]**2) <= 1.1:

            if center[0] > 0:

                center_moved_x = center[0] + dR * np.cos(np.pi/6)

            else:

                center_moved_x = center[0] - dR * np.cos(np.pi/6)

            if center[1] > 0:

                center_moved_y = center[1] + dR * np.sin(np.pi/6)

            else:

                center_moved_y = center[1] - dR * np.sin(np.pi/6)

            center_final =  mp.Vector3(center_moved_x, center_moved_y)
            
            geometry.append(mp.Prism(vertices_NN,
                                height=t,
                                center=center_final,
                                material=mp.Medium(index = 1))) #add the hexagons in the lattice 
            
            continue 
        
        geometry.append(mp.Prism(vertices,
                                height=t,
                                center=center_final,
                                material=mp.Medium(index = 1))) #add the hexagons in the lattice   
    return geometry
# %%

def calculate_PF(x):

    s1, s2, dR = x

    resolution = 2

    N= 27

    length= 30
    lengthz = 5

    a = 405e-9 # nm
    t = 180e-9 / a

    c = 3e8

    fcen = 350e12 * a / c                                     # 340 THz(inside band-gap), since a = 405 nm, and c = 3e8: f=freq*a/c
    df = 50e12 * 2 * a / c

    src = [mp.Source(src=mp.GaussianSource(fcen, fwidth=1),
                     center=mp.Vector3(x=0, y=0, z=0),
                     component=mp.Ex, 
                     amplitude=1.0),
      ]


    cell = mp.Vector3(length, length, lengthz)                      # 3D case

    dpml = 1

    pml_layers = [mp.PML(dpml)]

    geometry = [mp.Block(center=mp.Vector3(0,0,0), 
            size=mp.Vector3(length-2*dpml,length-2*dpml,t), 
            material=mp.Medium(index = 1))]                    # Block of air extending up to PML region 

    sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=src,
                    resolution=resolution,
                    force_complex_fields=True
                   )

    nfreq = 200
    pt = mp.Vector3(0,0)
    sim.run(mp.dft_ldos( fcen, df, nfreq),until_after_sources=mp.stop_when_fields_decayed(25,mp.Ex,pt,1e-8))

    gix0=sim.ldos_data

    geometry = create_structure(s1,s2,dR)

    sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=src,
                    resolution=resolution,
                    force_complex_fields=True
                   )
                   

    sim.run(mp.dft_ldos( fcen, df, nfreq),until=100)

    gix = sim.ldos_data

    PF_min = np.min(np.abs(np.real(gix)/np.real(gix0)))

    return PF_min


# %%

x0 = [1.0,1.0,0.0]
bounds = [(0.0,5.0),(0.0,5.0),(-1.0,1.0)]

results = opt.minimize(calculate_PF, x0, method="L-BFGS-B", bounds=bounds)

np.savetxt("optimization_results.txt",results)
# %%