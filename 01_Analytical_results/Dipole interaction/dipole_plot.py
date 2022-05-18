# %%

import numpy as np
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
from pyrsistent import field

plt.style.use("science")
mpl.rcParams.update({"font.size": 32})


def fields(freq, theta, phi, r):

    mu_0 = 1.25e-6
    c = 3e8
    p = -9.28e-24

    omega = 2 * np.pi * freq
    k = omega / c

    constant = mu_0 * omega ** 3 * p / (4 * np.pi * c)
    r_terms_theta = (1 / k * r) ** -3 - 1j * (1 / k * r) ** 2 - (1 / k * r)
    r_terms_r = 2 * (1 / k * r) ** 3 - 2j * (1 / k * r) ** 2

    E_theta = constant * np.sin(theta) * (np.exp(1j * k * r) * r_terms_theta)
    E_r = constant * np.cos(theta) * (np.exp(1j * k * r) * r_terms_r)

    return E_theta, E_r


def near_fields(freq, theta, r):

    mu_0 = 1.25e-6
    c = 3e8
    p = -9.28e-24

    omega = 2 * np.pi * freq
    k = omega / c

    constant = mu_0 * omega ** 3 * p / (4 * np.pi * c)
    r_terms_theta = (1 / (k * r)) ** -3  # - 1j*(1/kr)**2 - (1/kr)
    r_terms_r = 2 * (1 / (k * r)) ** 3  # - 2j*(1/kr)**2

    E_theta = constant * np.sin(theta) * (np.exp(1j * k * r) * r_terms_theta)
    E_r = constant * np.cos(theta) * (np.exp(1j * k * r) * r_terms_r)

    return E_theta, E_r


def far_fields(freq, theta, r):

    mu_0 = 1.25e-6
    c = 3e8
    p = -9.28e-24

    omega = 2 * np.pi * freq
    k = omega / c

    constant = mu_0 * omega ** 3 * p / (4 * np.pi * c)
    r_terms_theta = -(1 / (k * r))  # (1/kr)**-3 - 1j*(1/kr)**2 - (1/kr)
    r_terms_r = 0  # 2*(1/kr)**3 - 2j*(1/kr)**2

    E_theta = constant * np.sin(theta) * (np.exp(1j * k * r) * r_terms_theta)
    E_r = 0

    return E_theta, E_r


freq = 1e9  # Hz
c = 3e8
k = 2 * np.pi * freq / c
kx_close = np.linspace(-0.25, 0.25, 1000)
x_close = kx_close / k
kx_far = np.linspace(-1.0, 1.0, 1000)
x_far = kx_far / k

ky_close = kx_close
y_close = x_close
kz_close = kx_close
z_close = x_close
ky_far = kx_far
y_far = x_far
kz_far = kx_far
z_far = x_far
kr_black_ball = 0.1

field_values = np.zeros((len(x_close), len(x_close)), dtype=np.complex128)


x_close = 0.0
x_far = 0.0

for i in range(len(y_close)):
    for j in range(len(z_close)):
        r = np.sqrt(y_close[i] ** 2 + z_close[j] ** 2)
        theta = np.arccos(z_close[j] / r)
        if k * r < kr_black_ball:
            field_values[i, j] = 0.0
        else:
            field_theta, field_r = np.real(near_fields(freq, theta, r))
            field_values[i, j] = np.sqrt(field_theta ** 2 + field_r ** 2)

field_values = field_values / np.linalg.norm(field_values)

# %%
import matplotlib

fig, ax = plt.subplots(1, 2, figsize=(24, 12))
normalize = matplotlib.colors.Normalize(vmin=0, vmax=0.02)
ax[0].contourf(
    np.real(field_values).T,
    levels=25,
    cmap="inferno",
    extent=[ky_close[0], ky_close[-1], ky_close[0], ky_close[-1]],
)
ax[0].set_title("Near-field")
ax[0].set_xlabel("$ky$")
ax[0].set_ylabel("$kz$")
ax[1].set_xlabel("$ky$")
ax[1].set_ylabel("$kz$")
# %%
field_values_far = np.zeros((len(y_close), len(y_close)), dtype=np.complex128)

for i in range(len(y_far)):
    for j in range(len(z_far)):
        r = np.sqrt(y_far[i] ** 2 + z_far[j] ** 2)
        theta = np.arccos(z_far[j] / r)
        if k * r < kr_black_ball:
            field_values_far[i, j] = 0.0
        else:
            field_theta, field_r = np.real(far_fields(freq, theta, r))
            field_values_far[i, j] = np.sqrt(field_theta ** 2 + field_r ** 2)

field_values_far = field_values_far / np.linalg.norm(field_values)
# %%
# normalize = matplotlib.colors.Normalize(vmin=0, vmax=0.0002)
ax[1].contourf(
    np.real(field_values_far).T,
    levels=25,
    cmap="inferno",
    extent=[ky_far[0], ky_far[-1], ky_far[0], ky_far[-1]],
)  # ,norm=normalize)
ax[1].set_title("Far-field")

fig
# %%
fig.savefig("dipole_radiation.pdf")
# %%
