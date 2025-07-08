# Howru Sulfides

Howru is an automated method to determine Hubbard U and Jain corrections for a database of DFT and experimental formation energies.

# Features

Determine the optimial Hubbard U and Jain correction for 3d transition metal sulfides with reaction_grouping_test.py

# Installation

First construct a virtual environment

          python3 -m venv .venv 
          source .venv/bin/activate 
          pip install -r requirements.txt

and then the optimal Hubbard U and Jain Correction parameters can be determined with:

          python reaction_grouping_test.py


# References:

B. Donovan, A. West, A. Urban. "r2SCAN+rVV10+U parameterization for 3d transition metal sulfides for thermochemistry" J. Chem. Phys. (2025) ASAP
