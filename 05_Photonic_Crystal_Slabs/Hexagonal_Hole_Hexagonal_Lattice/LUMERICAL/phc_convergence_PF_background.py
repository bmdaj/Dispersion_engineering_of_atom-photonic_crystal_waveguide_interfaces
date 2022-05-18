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
def create_fsp_file(params, i, num):
    """This function creates a fsp. file  so  that it can be run in the
    background using the command line. The input parameters are:
     @ meshacc:  Parameter between 1 and 8.The mesh accuracy setting of 2 which corresponds to 10 mesh points
                per wavelength is considered reasonable for the FDTD method, and mesh accuracy 4 or 5 which
                corresponds to 18 or 22 mesh points per wavelength is considered high accuracy.
    @ time_factor: Time for FDTD calculation in LUMERICAL units, as in t = (time factor) * lX / c.
    @ nfreq: Number of points of frequency discretization.
    @ i : Dipole orientation, 0 = X, 1 = Y, 2 = Z.
    """
    #  -------------------------------------------------------------------------------------
    #  SIMULATION PARAMETERS
    #  -------------------------------------------------------------------------------------

    meshacc, timefactor, nfreq = params

    wSimVolXY = 6.0e-6  # The 2D PhC plane
    wSimVolZ = 1.0e-6  # propagation direction (okay it's also thickness direction)

    c = 3e8  # Speed of  light

    #  -------------------------------------------------------------------------------------
    # STRUCTURAL PARAMETERS
    #  -------------------------------------------------------------------------------------
    # MATERIAL

    nSiN = 1.997  # Refractive index of SiN
    tSiN = 180e-9  # Thickness of SiN layer

    # OTHER PARAMETERS

    wMarginXY = 0.5e-6  # Margin width where there will be no holes in the supercell
    nMarginIndex = 1.997  # Refractive index of Si3N4
    aPhC = 410e-9  # Lattice constant
    tTether = 170e-9
    dzDP = 0e-9  # Displacement of the dipole over Z-axis (if we cannot get the atoms in holes)

    # TO BE IMPLEMENTED: Adjustment of parameters of holes. Center + NN.

    dR0 = 90e-1
    dR1 = 0e-9
    dp1 = 45e-9

    # CENTERS OF HOLES:

    # Lattice vector parametrization

    a1 = aPhC * np.array([1, 0])
    a2 = aPhC * np.array([-0.5, 0.5 * np.sqrt(3)])
    a3 = aPhC * np.array([-0.5, -0.5 * np.sqrt(3)])

    # Prepare sim-vol layout
    nPlot = int(np.ceil((wSimVolXY - wMarginXY) / aPhC))
    XX = np.array(
        np.matmul(
            np.reshape(np.linspace(0, nPlot, nPlot + 1), (nPlot + 1, 1)),
            np.ones((1, nPlot + 1)),
        )
    )
    YY = np.array(
        np.matmul(
            np.ones((nPlot + 1, 1)),
            np.reshape(np.linspace(0, nPlot, nPlot + 1), (1, nPlot + 1)),
        )
    )

    px1 = np.reshape(a1[0] * XX + a2[0] * YY, [(nPlot + 1) ** 2, 1])
    py1 = np.reshape(a1[1] * XX + a2[1] * YY, [(nPlot + 1) ** 2, 1])

    px1 = np.vstack([px1, np.reshape(a3[0] * XX + a2[0] * YY, [(nPlot + 1) ** 2, 1])])
    py1 = np.vstack([py1, np.reshape(a3[1] * XX + a2[1] * YY, [(nPlot + 1) ** 2, 1])])

    px1 = np.vstack([px1, np.reshape(a1[0] * XX + a3[0] * YY, [(nPlot + 1) ** 2, 1])])
    py1 = np.vstack([py1, np.reshape(a1[1] * XX + a3[1] * YY, [(nPlot + 1) ** 2, 1])])

    px1 = np.reshape(px1, len(px1))
    py1 = np.reshape(py1, len(py1))

    # We can plot the center - Debugging purposes
    # plt.scatter(px1, py1)

    # INDEX SEARCH FOR HOLE MODIFICATION
    stol = 1e-9
    # Tolerance
    indH0 = np.argwhere(np.abs(px1 ** 2 + py1 ** 2) < stol ** 2)  # Center hole
    indH1 = np.argwhere(
        (np.abs(px1 ** 2 + py1 ** 2) > (aPhC - stol) ** 2)
        & (np.abs(px1 ** 2 + py1 ** 2) < (aPhC + stol) ** 2)
    )  # Nearest neighbor holes : i.e. first ring of holes
    indHN = np.argwhere(
        np.abs(px1 ** 2 + py1 ** 2) > (aPhC + stol) ** 2
    )  # Rest of holes

    #  -------------------------------------------------------------------------------------
    # TIME / FREQ DOMAIN INFORMATION
    #  -------------------------------------------------------------------------------------

    fcs_d1 = 335.116e12  # frequency of D1 line
    fcs_d2 = 351.726e12  # frequency of D2 line
    freq = 1.0 * fcs_d2 + 0.0 * fcs_d1
    bandwidth = 150e12  # This is giant. will it work?
    freqhighres = nfreq  # For more detailed plots

    # For use as input into the DipolePower() and SourcePower()
    fUse = (
        freq
        - 0.5 * bandwidth
        + bandwidth * np.linspace(0, (freqhighres - 1), freqhighres) / (freqhighres - 1)
    )

    #  -------------------------------------------------------------------------------------
    #  SETTING UP FDTD SIMULATION
    #  -------------------------------------------------------------------------------------

    fdtd = lumapi.FDTD(hide=True)
    fdtd.setresource("FDTD", 1, "processes", "8")  # We use 8 processes

    #  -------------------------------------------------------------------------------------
    # BUILDING THE STRUCTURE
    #  -------------------------------------------------------------------------------------

    # 1. We add the PhCS without holes

    rect = fdtd.addrect()
    rect["name"] = "Dplate"
    rect["index"] = nMarginIndex
    rect["x span"] = 2 * wSimVolXY + 2e-6
    rect["y span"] = 2 * wSimVolXY + 2e-6
    rect["z min"] = -0.5 * tSiN
    rect["z max"] = 0.5 * tSiN

    # 2. We create the hexagonal area where the holes sit

    base = fdtd.addpoly()
    base["name"] = "DHexSiNBase"
    base["index"] = nSiN
    base["z span"] = tSiN
    base["x"] = 0
    base["y"] = 0
    base["vertices"] = (
        nPlot
        * aPhC
        * np.array(
            [
                [1, 0.5, -0.5, -1, -0.5, 0.5],
                [
                    0,
                    0.8660254037844386,
                    0.8660254037844386,
                    0,
                    -0.8660254037844386,
                    -0.8660254037844386,
                ],
            ]
        )
    )

    # We get the parameters/shape for the holes
    if (aPhC - tTether) > 1e-9:
        RHole = (aPhC - tTether) / np.sqrt(3)
        # We get the 6 vertices of the holes
        thihole = np.pi / 3 * np.linspace(0, 5, 6) + np.pi / 6 * np.ones(6)
        pHoleX = RHole * np.cos(thihole)
        pHoleY = RHole * np.sin(thihole)

    # In the first approximation we will not modify the shape or position of any hole.
    # Center hole
    center_hole = fdtd.addpoly()
    center_hole["name"] = "DHexHole" + str(indH0[0])
    center_hole["index"] = 1.0
    center_hole["x"] = px1[indH0]  # x is phc propagation direction
    center_hole["y"] = py1[indH0]  #
    center_hole["z span"] = tSiN
    # vertMod = (RHole + dR0) * np.array([np.cos(thihole), np.sin(thihole)])
    center_hole["vertices"] = np.transpose([pHoleX, pHoleY])

    # First NN holes
    for j in range(1, len(indH1)):
        NN_hole = fdtd.addpoly()
        NN_hole["name"] = "DHexHole" + str(indH1[j])
        NN_hole["index"] = 1.0  # This is hole
        xUse = px1[indH1[j]]
        yUse = py1[indH1[j]]
        # pos_array = np.array([xUse, yUse])
        # posMod = pos_array + dp1 * pos_array / np.norm(pos_array)
        NN_hole["x"] = xUse  # x is phc propagation direction
        NN_hole["y"] = yUse  #
        NN_hole["z span"] = tSiN
        # vertMod = (RHole + dR1) * np.array([np.cos(thihole), np.sin(thihole)])
        NN_hole["vertices"] = np.transpose([pHoleX, pHoleY])

    # The 'normal' holes
    for j in range(1, len(indHN)):
        normal_hole = fdtd.addpoly()
        normal_hole["name"] = "DHexHole" + str(indHN[j])
        normal_hole["index"] = 1.0  # This is hole
        normal_hole["x"] = px1[indHN[j]]  # x is phc propagation direction
        normal_hole["y"] = py1[indHN[j]]  #
        normal_hole["z span"] = tSiN
        normal_hole["vertices"] = np.transpose([pHoleX, pHoleY])

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
    dipole_source["record local field"] = 1

    #  -------------------------------------------------------------------------------------
    #  SIMULATION AREA SETTINGS
    #  -------------------------------------------------------------------------------------

    FDTD_dom = fdtd.addfdtd()
    FDTD_dom["mesh accuracy"] = meshacc
    FDTD_dom["mesh refinement"] = "conformal variant 1"  # Will see how this goes
    FDTD_dom["simulation time"] = wSimVolXY / c * timefactor
    FDTD_dom["z min"] = -wSimVolZ
    FDTD_dom["z max"] = wSimVolZ
    FDTD_dom["x min"] = -wSimVolXY
    FDTD_dom["y min"] = -wSimVolXY
    FDTD_dom["x max"] = wSimVolXY
    FDTD_dom["y max"] = wSimVolXY

    # WARNING: Symmetries will be handled later

    #  -------------------------------------------------------------------------------------
    # MONITOR SETTINGS
    #  -------------------------------------------------------------------------------------

    # We put monitors at the dipole position.

    #  Field at dipole position

    # 1) Field on dipole source location for Green function

    profile = fdtd.addpower()
    profile["name"] = "profile"
    profile["monitor type"] = "2D Z-normal"
    profile["x"] = 0
    profile["y"] = 0
    profile["z"] = 0
    profile["x min"] = -wSimVolXY
    profile["x max"] = +wSimVolXY
    profile["x span"] = 2 * wSimVolXY
    profile["y min"] = -wSimVolXY
    profile["y max"] = +wSimVolXY
    profile["y span"] = 2 * wSimVolXY
    profile["override global monitor settings"] = 1
    profile["frequency points"] = freqhighres

    # We set power monitor for radiation losses

    # Parameters computed from convergence test

    L_rad = 10 * aPhC
    h_rad = 0.5e-6
    w_rad = 0.2e-6

    profile1 = fdtd.addpower()
    profile1["name"] = "profile_prime1"
    profile1["monitor type"] = "2D Z-normal"
    profile1["x"] = 0
    profile1["y"] = 0
    profile1["z"] = h_rad
    profile1["x min"] = -L_rad
    profile1["x max"] = +L_rad
    profile1["x span"] = 2 * L_rad
    profile1["y min"] = -L_rad
    profile1["y max"] = +L_rad
    profile1["y span"] = 2 * L_rad
    profile1["override global monitor settings"] = 1
    profile1["frequency points"] = freqhighres

    profile2 = fdtd.addpower()
    profile2["name"] = "profile_prime2"
    profile2["monitor type"] = "2D Y-normal"
    profile2["x"] = 0
    profile2["y"] = L_rad
    profile2["z"] = h_rad - 0.5 * w_rad
    profile2["x min"] = -L_rad
    profile2["x max"] = +L_rad
    profile2["x span"] = 2 * L_rad
    profile2["z min"] = h_rad - w_rad
    profile2["z max"] = h_rad
    profile2["z span"] = w_rad
    profile2["override global monitor settings"] = 1
    profile2["frequency points"] = freqhighres

    profile3 = fdtd.addpower()
    profile3["name"] = "profile_prime3"
    profile3["monitor type"] = "2D X-normal"
    profile3["x"] = L_rad
    profile3["y"] = 0
    profile3["z"] = h_rad - 0.5 * w_rad
    profile3["y min"] = -L_rad
    profile3["y max"] = L_rad
    profile3["y span"] = 2 * L_rad
    profile3["z min"] = h_rad - w_rad
    profile3["z max"] = h_rad
    profile3["z span"] = w_rad
    profile3["override global monitor settings"] = 1
    profile3["frequency points"] = freqhighres

    # For dipole
    thi = [90, 90, 0]
    phi = [0, 90, 0]

    print(i)

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
    fdtd.save(
        "lumerical-running-savefile" + "_convergence_phc_mesh_acc_" + str(int(num))
    )


# %%
# -----------------------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------------------

mesh_accs = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
time_factors = np.array([50.0, 100.0, 150.0, 200.0, 250, 300, 350])
nfreqs = np.array([100, 200, 300, 400, 500, 600, 700])

j = 0  # convergence test on X dipole orientation

for i in range(len(mesh_accs)):

    print("SIMULATION " + str(i) + " RUNNING...")

    mesh_acc = mesh_accs[i]
    time_factor = time_factors[i]
    nfreq = nfreqs[i]

    params = mesh_acc, time_factor, nfreq

    create_fsp_file(params, j, i)

for i in range(len(mesh_accs)):

    mesh_acc = mesh_accs[i]

    # We use 8 processes
    process = subprocess.Popen(
        [
            "nohup /opt/lumerical/v212/mpich2/nemesis/bin/mpiexec -n 8 /opt/lumerical/v212/bin/fdtd-engine-mpich2nem lumerical-running-savefile"
            + "_convergence_phc_mesh_acc_"
            + str(int(i))
            + ".fsp > output.txt </dev/null "
        ],
        shell=True,
    )  # this executes a shell command

    exit_code = process.wait()  # We wait till the process is finished
