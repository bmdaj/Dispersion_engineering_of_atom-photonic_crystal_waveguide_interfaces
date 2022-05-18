# %%
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(".")

from TMM_class import TMM

# %%

neff = 1.7
t = 200e-9
freqs = np.linspace(300, 380, 10000) * 1e12  # frequencies we want to scan

TMM_phc = TMM(freqs=freqs, na=neff, La=t, type="Film")

# %%

# %%

TMM_phc.create_matrix_structure()
t = TMM_phc.calculate_transmission()
r = TMM_phc.calculate_reflection()

# %%

fig, ax = plt.subplots(1, 2, figsize=(10, 6))

ax[0].plot(freqs, t)
ax[1].plot(freqs, r)


ax[0].set_xlabel("freq")
ax[0].set_ylabel("t")
ax[1].set_xlabel("freq")
ax[1].set_ylabel("r")

# %%
