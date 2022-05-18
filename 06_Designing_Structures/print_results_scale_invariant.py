def print_results(freq):

    multiplier = 0.3744245837513734  # a units
    t = 1.313032080984342  # a units
    tm_band_freq = 0.5161192277189162  # meep units
    midgap_freq = 0.47334057192751394  # meep units
    band_gap_width = 0.0855579324702136  # meep units

    lattice_constant_midgap = 3e8 * midgap_freq / (freq * 1e-9)  # in nm
    band_gap_width_midgap = band_gap_width * freq / (midgap_freq * 1e12)  # in THz
    lattice_constant_tm_band = 3e8 * tm_band_freq / (freq * 1e-9)  # in nm
    band_gap_width_tm = band_gap_width * freq / (tm_band_freq * 1e12)  # in THz

    print(
        "--------------------------------------------------------------------------------"
    )
    print("STRUCTURAL RESULTS for freq: " + str(freq * 1e-12) + " THz")
    print(
        "--------------------------------------------------------------------------------"
    )
    print("ALLIGNMENT WITH MIDGAP FREQUENCY: ")
    print(
        "--------------------------------------------------------------------------------"
    )
    print("Lattice constant: {} nm".format(lattice_constant_midgap))
    print("Hexagon size: {} nm".format(lattice_constant_midgap * multiplier))
    print("Slab thickness: {} nm".format(lattice_constant_midgap * t))
    print("Band gap width: {} THz".format(band_gap_width_midgap))
    print(
        "--------------------------------------------------------------------------------"
    )
    print("ALLIGNMENT WITH TM BAND LIMIT: ")
    print(
        "--------------------------------------------------------------------------------"
    )
    print("Lattice constant: {} nm".format(lattice_constant_tm_band))
    print("Hexagon size: {} nm".format(lattice_constant_tm_band * multiplier))
    print("Slab thickness: {} nm".format(lattice_constant_tm_band * t))
    print("Band gap width: {} THz".format(band_gap_width_tm))


# %%

print(
    "--------------------------------------------------------------------------------"
)
print("Cs D2 line: ")
print(
    "--------------------------------------------------------------------------------"
)
freq = 351.726e12
print_results(freq)
print(
    "--------------------------------------------------------------------------------"
)
print("Rb D1 line: ")
print(
    "--------------------------------------------------------------------------------"
)
freq = 384.23e12
print_results(freq)
print(
    "--------------------------------------------------------------------------------"
)
print("Rb D2 line: ")
print(
    "--------------------------------------------------------------------------------"
)
freq = 377.107e12
print_results(freq)
print(
    "--------------------------------------------------------------------------------"
)
print("He* line: ")
print(
    "--------------------------------------------------------------------------------"
)
freq = 276.817e12
print_results(freq)
print(
    "--------------------------------------------------------------------------------"
)
print("Sr closed transition: ")
print(
    "--------------------------------------------------------------------------------"
)
freq = 166.551e12
print_results(freq)
