# %%
# -----------------------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------------------
import sys, os
import subprocess
import time

sys.path.append("/opt/lumerical/v212/api/python/")  # Default windows lumapi path
sys.path.append("/opt/lumerical/v212/api/")
sys.path.append("/opt/lumerical/v212/bin/")
sys.path.append("/home/bmda/anaconda3/envs/pmp/lib/python3.9/site-packages/")
sys.path.append(os.path.dirname(__file__))  # Current directory

import lumapi
import numpy as np
import matplotlib.pyplot as plt

# %%
def create_fsp_file(params, i, sim_num):
    """This function creates a fsp. file  so  that it can be run in the
    background using the command line. The input parameters are:
     @ w_b: Width of the box (Y direction, perp. to waveguide).
    @ l_b: Lenght of the box (X direction, parallel to waveguide).
    @ h_b: Height of the box (Z direction, parallel to waveguide)
    @ i : Dipole orientation, 0 = X, 1 = Y, 2 = Z.
    """
    #  -------------------------------------------------------------------------------------
    #  SIMULATION PARAMETERS
    #  -------------------------------------------------------------------------------------

    meshacc = 3

    l_b, w_b, h_b = params

    wSimVolX = 16e-6  # The 1D PhC waveguide axis
    wSimVolY = 4.0e-6
    wSimVolZ = 4.0e-6

    c = 3e8  # Speed of  light

    #  -------------------------------------------------------------------------------------
    # STRUCTURAL PARAMETERS
    #  -------------------------------------------------------------------------------------
    # MATERIAL

    nSiN = 1.997  # The refractive index of Si3N4

    # OTHER PARAMETERS

    len = 13.8e-6  # The 1D PhC waveguide axis
    a = 335e-9  # Lattice constant of structure
    w = 335e-9  # Width of structure
    t = 200e-9  # Thickness of structure
    r = 0.5 * 116e-9  # Radius of holes in structure
    g = 250e-9  # Width of gap in structure

    dzDP = 0e-9  # Displacement of the dipole over Z-axis

    #  -------------------------------------------------------------------------------------
    # TIME / FREQ DOMAIN INFORMATION
    #  -------------------------------------------------------------------------------------

    freq = 350e12  # Choice for center frequency of pulse
    bandwidth = 250e12
    freqhighres = 400  # For more detailed plots

    # For use as input into the DipolePower() and SourcePower()
    fUse = (
        freq
        - 0.5 * bandwidth
        + bandwidth * np.linspace(0, (freqhighres - 1), freqhighres) / (freqhighres - 1)
    )
    timefactor = 100
    #  -------------------------------------------------------------------------------------
    #  SETTING UP FDTD SIMULATION
    #  -------------------------------------------------------------------------------------

    fdtd = lumapi.FDTD(hide=True)  # to not show GUI
    fdtd.setresource("FDTD", 1, "processes", "8")  # We use 8 processes
    #  -------------------------------------------------------------------------------------
    # BUILDING THE STRUCTURE
    #  -------------------------------------------------------------------------------------

    # 1. We add the first nanobeam

    rect1 = fdtd.addrect()
    rect1["name"] = "Dplate1"
    rect1["index"] = nSiN
    rect1["x span"] = 2 * len
    rect1["y span"] = w
    rect1["z min"] = -0.5 * t
    rect1["z max"] = 0.5 * t
    rect1["x"] = 0.0
    rect1["y"] = -0.5 * (g + w)
    rect1["z"] = 0.0

    # 2. We add the first nanobeam

    rect2 = fdtd.addrect()
    rect2["name"] = "Dplate2"
    rect2["index"] = nSiN
    rect2["x span"] = 2 * len
    rect2["y span"] = w
    rect2["z min"] = -0.5 * t
    rect2["z max"] = 0.5 * t
    rect2["x"] = 0.0
    rect2["y"] = +0.5 * (g + w)
    rect2["z"] = 0.0

    # 3. We add holes

    Nholes = int(2 * len // a - 1)
    print("Number of holes is:", Nholes)
    for j in range(-Nholes, Nholes + 1):
        circ1 = fdtd.addcircle()
        circ1["name"] = "Circ_1_" + str(j)
        circ1["index"] = 1
        circ1["radius"] = r
        circ1["z span"] = t
        circ1["x"] = (j - 0.5) * a
        circ1["y"] = -0.5 * (g + w)
        circ1["z"] = 0
        circ1["make ellipsoid"] = 1
        circ1["radius 2"] = r
        circ2 = fdtd.addcircle()
        circ2["name"] = "Circ_2_" + str(j)
        circ2["index"] = 1
        circ2["radius"] = r
        circ2["z span"] = t
        circ2["x"] = (j - 0.5) * a
        circ2["y"] = 0.5 * (g + w)
        circ2["z"] = 0
        circ2["make ellipsoid"] = 1
        circ2["radius 2"] = r

    #  -------------------------------------------------------------------------------------
    # SOURCE SETTINGS
    #  -------------------------------------------------------------------------------------

    dipole_source = fdtd.adddipole()
    dipole_source["name"] = "DSource"
    dipole_source["x"] = 0
    dipole_source["y"] = 0
    dipole_source[
        "z"
    ] = dzDP  # This is how much the dipole is displaced on the Z axis, if it cannot be trapped correctly
    dipole_source["set frequency"] = 1
    dipole_source["override global source settings"] = True
    dipole_source["center frequency"] = freq
    dipole_source["frequency span"] = bandwidth
    dipole_source["optimize for short pulse"] = True
    # Normalize for injected power source:
    fdtd.cwnorm(1)  # Normlalized to the first active source
    # fdtd.nonorm()
    # WARNING: Polarization will be set later in the loop

    dipole_source[
        "record local field"
    ] = 1  # Important to be able to calculate the Green's function

    #  -------------------------------------------------------------------------------------
    #  SIMULATION AREA SETTINGS
    #  -------------------------------------------------------------------------------------

    FDTD_dom = fdtd.addfdtd()
    FDTD_dom["mesh accuracy"] = meshacc
    FDTD_dom["mesh refinement"] = "conformal variant 1"
    FDTD_dom["simulation time"] = wSimVolX / c * timefactor
    FDTD_dom["z min"] = -wSimVolZ
    FDTD_dom["z max"] = wSimVolZ
    FDTD_dom["x min"] = -wSimVolX
    FDTD_dom["y min"] = -wSimVolY
    FDTD_dom["x max"] = wSimVolX
    FDTD_dom["y max"] = wSimVolY
    FDTD_dom["auto shutoff min"] = 1e-99  # so it finishes the simulations

    # We add another mesh in the center of dipole, to be able to get Green's functions accurately

    x = fdtd.getresult("FDTD", "x")
    dx = x[1] - x[0]
    dx_new = 0.75 * dx  # This should be smaller than previous dx
    center_mesh = fdtd.addmesh()
    center_mesh["x"] = 0
    center_mesh["y"] = 0
    center_mesh["z"] = 0
    center_mesh["x min"] = -dx_new
    center_mesh["x max"] = +dx_new
    center_mesh["x span"] = 2 * dx_new

    # WARNING: Symmetries will be handled later

    # We set monitors

    profile1 = fdtd.addpower()
    profile1["name"] = "profile_prime1"
    profile1["monitor type"] = "2D Z-normal"
    profile1["x"] = 0
    profile1["y"] = 0
    profile1["z"] = h_b
    profile1["x min"] = -l_b
    profile1["x max"] = +l_b
    profile1["x span"] = 2 * l_b
    profile1["y min"] = -w_b
    profile1["y max"] = +w_b
    profile1["y span"] = 2 * w_b
    profile1["override global monitor settings"] = 1
    profile1["frequency points"] = freqhighres

    profile3 = fdtd.addpower()
    profile3["name"] = "profile_prime3"
    profile3["monitor type"] = "2D Y-normal"
    profile3["x"] = 0
    profile3["y"] = w_b
    profile3["z"] = 0
    profile3["x min"] = -l_b
    profile3["x max"] = +l_b
    profile3["x span"] = 2 * l_b
    profile3["z min"] = -h_b
    profile3["z max"] = +h_b
    profile3["z span"] = 2 * h_b
    profile3["override global monitor settings"] = 1
    profile3["frequency points"] = freqhighres

    # For dipole
    thi = [90, 90, 0]
    phi = [0, 90, 0]

    fdtd.select("DSource")
    fdtd.set("theta", thi[i])
    fdtd.set("phi", phi[i])

    # Handle symmetry
    fdtd.select("FDTD")

    fdtd.set("x min bc", "Symmetric")
    fdtd.set("y min bc", "Symmetric")
    fdtd.set("z min bc", "Symmetric")

    if i == 0:
        fdtd.set("x min bc", "Anti-Symmetric")
    elif i == 1:
        fdtd.set("y min bc", "Anti-Symmetric")
    else:  # i == 2
        fdtd.set("z min bc", "Anti-Symmetric")
    # Make Savefile
    fdtd.save("lumerical-running-savefile" + "_convergence_" + str(sim_num))


# %%
# -----------------------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------------------

a = 335e-9  # Lattice constant of structure

l_bs = np.array([5, 10, 20, 30, 40]) * a
w_bs = np.array([0.5, 1, 2, 4, 8]) * a
h_bs = np.array([0.25, 0.5, 1, 2, 4]) * a

nfreq = 400  # from  last convergence test

j = 0  # convergence test on X dipole orientation


for i in range(len(l_bs)):

    print("SIMULATION " + str(i) + " RUNNING...")

    l_b = l_bs[i]
    w_b = w_bs[i]
    h_b = h_bs[i]

    params = l_b, w_b, h_b

    create_fsp_file(params, i=j, sim_num=i)

for i in range(len(l_bs)):

    l_b = l_bs[i]
    w_b = w_bs[i]
    h_b = h_bs[i]

    params = l_b, w_b, h_b

    # We use 8 processes

    process = subprocess.Popen(
        [
            "nohup /opt/lumerical/v212/mpich2/nemesis/bin/mpiexec -n 8 /opt/lumerical/v212/bin/fdtd-engine-mpich2nem lumerical-running-savefile"
            + "_convergence_"
            + str(i)
            + ".fsp > output.txt </dev/null "
        ],
        shell=True,
    )  # this executes a shell command

    exit_code = process.wait()  # We wait till the process is finished
