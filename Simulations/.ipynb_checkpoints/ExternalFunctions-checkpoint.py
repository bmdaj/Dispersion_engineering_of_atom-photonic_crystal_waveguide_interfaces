# ---------------------------------------------------------------------------------------------#
# IN THIS MODULE WE DEFINE EXTERNAL FUNCTIONS TO MAKE THE REST OF THE CODE MORE MODULAR
# ---------------------------------------------------------------------------------------------#

import numpy as np
from meep import mpb
import meep as mp
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------#
# GEOMETRY FUNCTIONS AND ROUTINES
# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#
# This function describes the geometry of the APCW structure
# ---------------------------------------------------------------------------------------------#

def geom_apcw (params):
    
    # Extraction of parameters
    
    dielec_const, period, amp, width, thickness, gap = params
    
    def  geometry(Vector3):
        
        array3 = np.array(Vector3)
        x = array3 [0]
        y = array3 [1]
        z = array3 [2]
    
        if y >= 0.5*gap and amp*np.cos(2*np.pi*(x)) + 0.5*gap+width>= y and 0.5 *thickness > np.abs(z):
            eps = mp.Medium(index = dielec_const) 
        
        elif y <= -0.5*width and -amp*np.cos(2*np.pi*(x)) - 0.5*gap-width <= y and 0.5 *thickness > np.abs(z):
            eps = mp.Medium(index = dielec_const) 
        else:
            eps = mp.Medium(epsilon = 1)
        
        if  width-amp+0.5*gap  >= np.abs(y) >= 0.5*gap and 0.5 *thickness > np.abs(z) :
            eps = mp.Medium(index = dielec_const) 
            
        return eps

    return geometry

# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------#
# CALCULATION FUNCTIONS AND ROUTINES
# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#
# This function calculates the electric field
# ---------------------------------------------------------------------------------------------#

def calculate_efield(sim_params, num_period, component):
    
    geometry_lattice,  k_point, default_material, resolution, num_bands = sim_params
    
    ms = mpb.ModeSolver(geometry_lattice=geometry_lattice,
                    k_points=[k_point],
                    default_material=default_material,
                    resolution=resolution,
                    num_bands=num_bands);
    
    efields = []

    def get_efields(ms, band):
        efields.append(ms.get_efield(band, bloch_phase=True))

    ms.run(mpb.output_at_kpoint(k_point, mpb.fix_efield_phase,
          get_efields))

    # Create an MPBData instance to transform the dfields
    md = mpb.MPBData(rectify=True, resolution=resolution, periods=num_period)

    converted = []
    for f in tqdm(efields):       
        
        if component == 'X':
            f = f[..., 0, 0]
            
        if component == 'Y':
            f = f[..., 0, 1]
            
        if component == 'Z':
            f = f[..., 0, 2]
            
        converted.append(md.convert(f))
    
    eps = ms.get_epsilon()
    converted_eps = md.convert(eps).T
        
    return converted, converted_eps

# ---------------------------------------------------------------------------------------------#
# This function gives back the simulation domain length 
# ---------------------------------------------------------------------------------------------#

def  simulation_domain(a, eps, resolution):
    
    domain = np.array(np.shape(eps))
    distance_conversion = a / resolution
    domain_limits = domain*distance_conversion
    
    print('The simulation domain lengths are:', domain_limits.astype(int))
    
    return domain_limits.astype(int)

# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------#
# PLOTTING FUNCTIONS AND ROUTINES
# ---------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------#
# This function plots the band diagram in THZ
# ---------------------------------------------------------------------------------------------#

def plot_bands_1D_THZ (a, num_bands, files, xlims, ylims):
    
    from scipy.interpolate import make_interp_spline, BSpline

    fig, ax = plt.subplots(figsize=(14,10));

    light_speed = 3e8
    
    f1, f2 = files

    for i in range(1,num_bands):
        xnew = np.linspace((f1[1:,1]*2).min(), (f1[1:,1]*2).max(), 300) 
        spl1 = make_interp_spline(f1[1:,1]*2, f1[1:,-i]*light_speed/(a*10**12), k=3)  # type: BSpline
        spl2 = make_interp_spline(f1[1:,1]*2, f2[1:,-i]*light_speed/(a*10**12), k=3)  # type: BSpline
        TE_mode_smooth = spl1(xnew)
        TM_mode_smooth = spl2(xnew)
        if i ==1:
            ax.plot(xnew, TE_mode_smooth, c='orange', linestyle='dashed', label='TE modes') 
            ax.plot(xnew, TM_mode_smooth, c='blue', linestyle='dashed', label='TM modes') 
        else:
            ax.plot(xnew, TE_mode_smooth, c='orange', linestyle='dashed')
            ax.plot(xnew, TM_mode_smooth, c='blue', linestyle='dashed') 
    ax.fill_between(xnew, np.ones(np.shape(xnew))*f1[-1,-1]*light_speed/(a*10**12),  np.ones(np.shape(xnew))*f1[-1,-2]*light_speed/(a*10**12), color='orange', alpha=0.2, label='TE gap') 
    ax.fill_between(xnew, np.ones(np.shape(xnew))*f2[-1,-1]*light_speed/(a*10**12), np.ones(np.shape(xnew))*f2[-1,-2]*light_speed/(a*10**12), color='blue', alpha=0.2, label='TM gap') 
    ax.plot(np.linspace(0,1,100),np.linspace(0,1,100)*light_speed/(2*a*10**12), c='r', label ='Light line')
    ax.fill_between(np.linspace(0,1,100), np.linspace(0,1,100)*light_speed/(2*a*10**12), 1*np.ones(100)*light_speed/(2*a*10**12), color='purple', alpha=0.8, label='Light cone') 

    ax.set_title('Dispersion diagram for the APCW waveguide model')
    ax.set_xlabel("Normalized $k_x a / \\pi$")
    ax.set_ylabel("Frequency $\\nu$ (THz)")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.legend()
    return fig

# ---------------------------------------------------------------------------------------------#
# This function plots the dielectric material distribution in YZ plane
# ---------------------------------------------------------------------------------------------#

def plot_epsilon_YZ (sim_params, a, num_period, ylim,  zlim, title):
    
    geometry_lattice,  k_point, default_material, resolution, num_bands = sim_params
    
    ms = mpb.ModeSolver(geometry_lattice=geometry_lattice,
                k_points=[k_point],
                default_material=default_material,
                resolution=resolution,
                num_bands=num_bands);
    ms.run();
    
    eps = ms.get_epsilon()
    
    md = mpb.MPBData(rectify=True, periods=num_period, resolution=resolution)
    
    converted_eps = md.convert(eps).T
    
    fig, ax = plt.subplots(figsize=(12,10))
    ax.set_title(title)
    
    x_half_index = int( np.shape(converted_eps)[0] // 2 )
    
    distance_conversion = resolution / a
    
    ymin, ymax = ylim
    zmin, zmax = zlim
    
    y_offset = int(np.array(np.shape(converted_eps))[1]//2)
    z_offset = int(np.array(np.shape(converted_eps))[2]//2)
    
    ymin_index = int( ymin * distance_conversion ) + y_offset
    ymax_index = int( ymax * distance_conversion ) + y_offset
    zmin_index = int( zmin * distance_conversion ) + z_offset
    zmax_index = int( zmax * distance_conversion ) + z_offset
    
    extent = [zmin , zmax, ymin , ymax]
    
    ax.imshow(converted_eps[x_half_index,ymin_index:ymax_index,zmin_index:zmax_index], extent = extent, interpolation='spline36', cmap='binary')
    ax.set_ylabel('Y distance')
    ax.set_xlabel('Z distance')
    return fig, ax

# ---------------------------------------------------------------------------------------------#
# This function plots a 2D field in the YZ projection
# ---------------------------------------------------------------------------------------------#

def plot_field_2D_YZ (eps, field, a, resolution, num_band, ylim,  zlim, title):
    
    fig, ax = plt.subplots(figsize=(12,10))
    ax.set_title(title)
    
    x_half_index = int( np.shape(eps)[0] // 2 )
    
    distance_conversion = resolution / a
    
    ymin, ymax = ylim
    zmin, zmax = zlim
    
    y_offset = int(np.array(np.shape(eps))[1]//2)
    z_offset = int(np.array(np.shape(eps))[2]//2)
    
    ymin_index = int( ymin * distance_conversion ) + y_offset
    ymax_index = int( ymax * distance_conversion ) + y_offset
    zmin_index = int( zmin * distance_conversion ) + z_offset
    zmax_index = int( zmax * distance_conversion ) + z_offset
    
    extent = [zmin , zmax, ymin , ymax]
        
    y_range = np.linspace(ymin,ymax,ymax_index-ymin_index) 
    z_range = np.linspace(zmin,zmax,zmax_index-zmin_index)
    
    ax.contour(eps[x_half_index,ymin_index:ymax_index,zmin_index:zmax_index], extent=extent, cmap='binary')
    posd1 = ax.imshow(np.real(field[num_band].T[ymin_index:ymax_index,zmin_index:zmax_index]), extent=extent, interpolation='spline36', cmap='plasma')
    ax.set_ylabel('Y distance')
    ax.set_xlabel('Z distance')
    fig.colorbar(posd1,ax=ax)
    
    return fig, ax