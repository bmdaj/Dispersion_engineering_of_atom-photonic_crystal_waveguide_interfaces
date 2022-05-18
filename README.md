# Dispersion engineering of atom-nanophotonic crystal waveguide interfaces

This repository serves as a compilation of the associated to Beñat Martinez de Aguirre's MSc. Thesis in Computational Physics at the NBI. Using analytical methods, MEEP and Lumerical we study and optimize the dispersion properties of photonic crystals as well as the emission properties of atoms close to these structures.

Citing: Martinez de Aguirre, B. *Dispersion engineering of atom-nanophotonic crystal waveguide interfaces*. MSc. Thesis (University of Copenhagen, 2022)

## Structure of the repository

In this repository one can find the necessary files to reproduce the all of the results in the thesis. The structure of the repository works its way up in levels of software difficultya and has the following entries:

- **Example** (Folder 0): Brief overview of how to calculate the main Figures of Merit (FOMs) in the case of the hexagonal holed hexagonal lattice, with an explanation on how to run a Lumerical FDTD simulation through its Python API. This example goes hand in hand with the example in Appendix C of the thesis.

- **Analytical results** (Folder 1): Calculations for the TMM formalism (Chapters 2 & 3) and dipole interactions (Chapter 3).

- **Dispersion and emission properties of photonic crystals** (Folders 2-5): Includes all the relevant calculations for the photonic crystals from the reviewed literature (Chapter 4), including the double nanobeam and the photonic crystal slabs. It makes use of MPB to calculate band-diagrams and mode profiles and uses Lumerical and MEPP for calculating the emission properties from FDTD calculations.

- **New designs for enhanced FOMs** (Folder 6): Novel calculations for the designs leading to improved FOMs. This includes the calculations for the different atoms and the 

## Contact

For more information or comments please feel free to reach out at: <bmdaj13@gmail.com>
