import numpy as np
import matplotlib.pyplot as plt

# %%
### Latex style
import matplotlib as mpl

plt.style.use("science")
mpl.rcParams.update({"font.size": 34})
figwidth = 20
figheight = 14
fig, ax = plt.subplots(3, 1, figsize=(figwidth, figheight))
plt.rcParams["axes.titlepad"] = 25
##################

# time variable
npoints = 40000
t = np.linspace(0, 1.01, npoints, endpoint=True)
print("wait a few seconds..")

## case 1
## only dissipative coupling

p0 = 1.0  # unit of dipole strength
Gp = 0.0  # Gamma_prime
G11 = 3  # Gamma_11
G12 = 3  # Gamma_12
omega0 = 400.0  # arbitrary optical angular frequency
J11 = 0  # J_11
J12 = 0  # J_12
p1 = np.real(
    p0
    / 2.0
    * np.exp(-t * (Gp + G11 - G12))
    * np.exp(1j * (J11 - omega0) * t)
    * (np.exp(-2 * t * G12) * np.exp(1j * t * J12) + np.exp(-1j * t * J12))
)
p2 = np.real(
    p0
    / 2.0
    * np.exp(-t * (Gp + G11 - G12))
    * np.exp(1j * (J11 - omega0) * t)
    * (np.exp(-2 * t * G12) * np.exp(1j * t * J12) - np.exp(-1j * t * J12))
)

ax[0].plot(t, p1, label="p$_1$")
ax[0].plot(t, p2, label="p$_2$", color="red", lw=1)
ax[0].set_ylabel("p(t)")
ax[0].set_xlim([0, t[-1]])
ax[0].set_xticks([0, 1])
ax[0].set_yticks([-1, 0, 1])
ax[0].set_title(
    "$\omega_0 = 400\Gamma_0$, $J_{11} = J_{12} = 0$, $\Gamma_{11} = \Gamma_{12} = 3 \Gamma_0$"
)

## case 2
## pure dispersive coupling

p0 = 1.0
Gp = 0.0
G11 = 0
G12 = 0
J11 = 6
J12 = 6
p1 = np.real(
    p0
    / 2.0
    * np.exp(-t * (Gp + G11 - G12))
    * np.exp(1j * (J11 - omega0) * t)
    * (np.exp(-2 * t * G12) * np.exp(1j * t * J12) + np.exp(-1j * t * J12))
)
p2 = np.real(
    p0
    / 2.0
    * np.exp(-t * (Gp + G11 - G12))
    * np.exp(1j * (J11 - omega0) * t)
    * (np.exp(-2 * t * G12) * np.exp(1j * t * J12) - np.exp(-1j * t * J12))
)

ax[1].plot(t, p1, label="p$_1$")
ax[1].plot(t, p2, label="p$_2$", color="red", lw=1)
ax[1].set_ylabel("p(t)")
ax[1].set_xlim([0, t[-1]])
ax[1].set_xticks([0, 1])
ax[1].set_yticks([-1, 0, 1])
ax[1].set_title(
    "$\omega_0 = 400\Gamma_0$, $J_{11} = J_{12} = 6\Gamma_0$, $\Gamma_{11} = \Gamma_{12} = 0$"
)


## case 3
## dispersive coupling in 1d waveguide

p0 = 1.0
Gp = 0.0
G11 = 3
G12 = 0
J11 = 6
J12 = 6
p1 = np.real(
    p0
    / 2.0
    * np.exp(-t * (Gp + G11 - G12))
    * np.exp(1j * (J11 - omega0) * t)
    * (np.exp(-2 * t * G12) * np.exp(1j * t * J12) + np.exp(-1j * t * J12))
)
p2 = np.real(
    p0
    / 2.0
    * np.exp(-t * (Gp + G11 - G12))
    * np.exp(1j * (J11 - omega0) * t)
    * (np.exp(-2 * t * G12) * np.exp(1j * t * J12) - np.exp(-1j * t * J12))
)

ax[2].plot(t, p1, label="p$_1$")
ax[2].plot(t, p2, label="p$_2$", color="red", lw=1)
ax[2].set_xlabel("t  $(\Gamma_0)$")
ax[2].set_ylabel("p(t)")
ax[2].set_xlim([0, t[-1]])
ax[2].set_xticks([0, 1])
ax[2].set_yticks([-1, 0, 1])
ax[2].set_title(
    "$\omega_0 = 400\Gamma_0$, $J_{11} = J_{12} = 6\Gamma_0$, $\Gamma_{11} = 3, \Gamma_{12} = 0$"
)
ax[2].legend(loc="best", ncol=2, frameon=True)

## save figure
# dpi = 300
plt.tight_layout()
plt.savefig("dipoles.pdf", format="pdf", bbox_inches="tight")
plt.show()
# %%
