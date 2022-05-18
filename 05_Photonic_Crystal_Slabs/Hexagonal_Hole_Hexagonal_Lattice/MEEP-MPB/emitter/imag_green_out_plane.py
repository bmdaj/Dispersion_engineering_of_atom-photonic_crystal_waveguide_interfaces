import meep as mp 
import numpy as np
import matplotlib.pyplot as plt
from hexalattice.hexalattice import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


def imag_green_broadband (position):
    
    atom_pos = position
    
    #-----------------------------------------------------------------------------------------------------#
    # Simulation parameters
    #-----------------------------------------------------------------------------------------------------#
    
    print('-----------------------------------------------------------------------------------------------')
    print('SETTING PARAMETERS ...')
    print('-----------------------------------------------------------------------------------------------')
    
    a = 405e-9 # nm                                              # lattice constant
    t = 180e-9 / a                                               # thickness of slab

    c = 3e8

    fcen = 350e12 * a / c                                        # normalized frequency (MEEP units) 
    df = 50e12 * 2 * a / c                                       # spectral width 
    nfreq = 500                                                  # number of frequency points
    
    #-----------------------------------------------------------------------------------------------------#
    # Simulation space
    #-----------------------------------------------------------------------------------------------------#
    
    resolution = 10                                              # 10 pixels per unit a 

    length = 30                                                  # Length of X & Y dimensions
    lengthz = 5                                                  # Length of Z dimension   

    cell = mp.Vector3(length, length, lengthz)                   # Simulation volume

    dpml = 1                                                     # PML layer thickness
    pml_layers = [mp.PML(dpml)]                                  # PML layer
    
    N = 27                                                       # Number of holes
    
    def free_space_geometry():
        
        geometry = [mp.Block(center=mp.Vector3(0,0,0), 
                    size=mp.Vector3(length-2*dpml,length-2*dpml,t), 
                    material=mp.Medium(index = 1))]
        
        return geometry
    
    def PC_geometry(N):
        
        geometry = [mp.Block(center=mp.Vector3(0,0,0), 
                    size=mp.Vector3(length-2*dpml,length-2*dpml,t), 
                    material=mp.Medium(index = 2))]              # Block of SiN extending up to PML region
        
        # A hexagonal prism defined by six vertices centered on the origin
        # of material crystalline silicon (from the materials library)

        multiplier = (1-t)*0.5

        vertices = [mp.Vector3(0,multiplier*2/np.sqrt(3)),
                    mp.Vector3(multiplier*1,multiplier*1/np.sqrt(3)),
                    mp.Vector3(multiplier*1,-multiplier*1/np.sqrt(3)),
                    mp.Vector3(0,multiplier*-2/np.sqrt(3)),
                    mp.Vector3(-multiplier*1,-multiplier*1/np.sqrt(3)),
                    mp.Vector3(-multiplier*1,multiplier*1/np.sqrt(3))]

        hex_centers, _ = create_hex_grid(nx=N, ny=N, crop_circ=N//2)

        for center in hex_centers: 
            center_final = mp.Vector3(center[0], center[1])
            geometry.append(mp.Prism(vertices,
                                     height=t,
                                     center=center_final,
                                     material=mp.Medium(index = 1))) #add the hexagons in the lattice
        
        return geometry
    
    #-----------------------------------------------------------------------------------------------------#
    # RUN SIMULATIONS
    #-----------------------------------------------------------------------------------------------------#
    
    def run(component):
        
        src = [mp.Source(src=mp.GaussianSource(fcen, fwidth=0.5),
                     center=atom_pos,
                     component=component, 
                     amplitude=1.0),
          ]                                                          # broadband Gaussian source
    
        # FREE SPACE CALCULATION
    
        geometry = free_space_geometry()                             # Free space geometry
    
        sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=src,
                        resolution=resolution,
                        force_complex_fields=True,
                        )                                            # simulation set up
    
        pt = atom_pos
        sim.run(mp.dft_ldos( fcen, df, nfreq),until_after_sources=mp.stop_when_fields_decayed(25,component,pt,1e-8))
        gi0=sim.ldos_data
        
    
        # PHOTONIC CRYSTAL CALCULATION
        
        geometry = PC_geometry(N)                                    # PC geometry
    
        sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=src,
                        resolution=resolution,
                        force_complex_fields=True,
                        )                                            # simulation set up
        
        pt = atom_pos
        sim.run(mp.dft_ldos(fcen, df, nfreq),until=750)
        gi=sim.ldos_data    
        
        
        return gi, gi0
    
    #-----------------------------------------------------------------------------------------------------#
    # Calculation for X direction
    #-----------------------------------------------------------------------------------------------------#
    
    print('-----------------------------------------------------------------------------------------------')
    print('CALCULATION FOR X DIRECTION ...')
    print('-----------------------------------------------------------------------------------------------')
    
    gix, gix0 = run(mp.Ex)
    
    #-----------------------------------------------------------------------------------------------------#
    # Calculation for Y direction
    #-----------------------------------------------------------------------------------------------------#
    
    print('-----------------------------------------------------------------------------------------------')
    print('CALCULATION FOR Y DIRECTION ...')
    print('-----------------------------------------------------------------------------------------------')
    
    giy, giy0 = run(mp.Ey)
    
    #-----------------------------------------------------------------------------------------------------#
    # Calculation for Z direction
    #-----------------------------------------------------------------------------------------------------#
    
    print('-----------------------------------------------------------------------------------------------')
    print('CALCULATION FOR Z DIRECTION ...')
    print('-----------------------------------------------------------------------------------------------')
    
    giz, giz0 = run(mp.Ez)
    
    return np.real(gix)/np.real(gix0), np.real(giy)/np.real(giy0), np.real(giz)/np.real(giz0)


a = 405 # nm                                              # lattice constant

x_distances = [200] #in nanometers


for i in range(len(x_distances)):
    
    position = mp.Vector3(x_distances[i] / a,0.0,0.0)     #we place the atom at x distance, 0.0 is the center in MEEP units
    
    green_tensor = imag_green_broadband (position)
    
    np.savetxt(str(x_distances[i])+'_nm_x_axis_green_tensor.txt',green_tensor)
