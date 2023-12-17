"""
Collection of utility functions.

"""

import itertools
import pymatgen
from pymatgen.apps.borg.hive import VaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen


__date__ = "2023-12-17"
__version__ = "0.1"


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    """
    xs = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(xs, n) for n in range(len(xs)+1))


def collect_entries_vasp(datapath, ncores=1, outfile="entries.json",
                         inc_structure=False, parameters=None,
                         data=None, verbose=False):
    """
    Get entries (i.e., compositions and energies) from local VASP
    calculations by searching through a directory tree.

    Arguments:
      datapath (str)    One or more root path(s) for data assimilation
      ncores (int)      Number of cores to use
      outfile (str)     File for saving collected entries
      inc_structure     If True, include atomic structures in entry
      parameters (list) Input file parameters to keep track of.
      data (list)       Additional output data that should be retrieved
                        from the VASP calculations.  Per default, the
                        'filename' is kept track of.
      verbose (bool)    If true, write number of found entries to stdout

    """
    # Always include the file name
    if data is None:
        data = ['filename']
    else:
        data.append('filename')

    drone = VaspToComputedEntryDrone(inc_structure=inc_structure,
                                     parameters=parameters, data=data)

    if isinstance(datapath, (list, tuple)):
        local_entries = []
        for path in datapath:
            queen = BorgQueen(drone, rootpath=path,
                              number_of_drones=ncores)
            local_entries += queen.get_data()
        queen._data = local_entries
    else:
        queen = BorgQueen(drone, rootpath=datapath,
                          number_of_drones=ncores)
        local_entries = queen.get_data()
    queen.save_data(outfile)

    # filter out invalid entries
    real_local_entries = []
    for entry in local_entries:
        if entry is not None:
            real_local_entries.append(entry)

    if verbose:
        print("{} local entries found.".format(len(real_local_entries)))
        print("Saving to file: {}".format(outfile))

    return real_local_entries


def load_entries_json(entryfiles, verbose=False):
    """
    Load VASP calculation entries from a json file.

    Arguments:
      entryfiles (str)  Path to a JSON file containing entries,
                        or list of paths to several such files
      verbose (bool)    If true, print additional info

    """

    drone = VaspToComputedEntryDrone()

    queen = BorgQueen(drone)
    if isinstance(entryfiles, (list, tuple)):
        local_entries = []
        for fname in entryfiles:
            queen.load_data(fname)
            local_entries += queen.get_data()
        queen._data = local_entries
    else:
        queen.load_data(entryfiles)
        local_entries = queen.get_data()

    # filter out invalid entries
    real_local_entries = []
    for entry in local_entries:
        if entry is not None:
            real_local_entries.append(entry)

    if verbose:
        print("{} entries loaded from file(s): {}".format(
            len(real_local_entries), entryfiles))

    return real_local_entries


import csv
import re
from math import gcd
from functools import reduce

def parse_formula(formula):
    """
    Parse the chemical formula into elements and their counts.
    
    Arguments:

    formula (str) chemical formula that will be parsed
    """
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    return {elem: int(count) if count else 1 for elem, count in elements}

def calculate_gcd(numbers):
    """
    Calculate the greatest common divisor of a list of numbers.
    
    Arguments:
    numbers (list) input of numbers that will then return the GCD
    """
    return reduce(gcd, numbers)

def empirical_formula(compound):
    """
    Calculate the empirical formula and the number of formula units.

    Arguments:
    compound (str): The chemical formula, e.g., 'Fe4S8'
    """
    elements_counts = parse_formula(compound)
    counts = list(elements_counts.values())

    # Find the greatest common divisor for the counts
    gcd_counts = calculate_gcd(counts)

    # Divide all counts by the gcd to get the simplest ratio
    simplest_ratio = {element: count // gcd_counts for element, count in elements_counts.items()}

    # Construct the empirical formula
    empirical_formula = ''.join(f"{element}{(count if count > 1 else '')}" for element, count in simplest_ratio.items())

    # Number of formula units is the sum of the counts in the empirical formula
    num_units = sum(simplest_ratio.values())

    return empirical_formula, num_units



def create_csv_from_entries(entries, csv_file, exp_data_csv):
    """
    Create a CSV file from VASP calculation entries.
    
    Arguments:
    entries (list): List of entries from load_entries_json function
    csv_file (str): Path to save the CSV file
    exp_data_csv (str): Path to the CSV file with experimental formation energies
    """
    # Read experimental formation energies into a dictionary
    exp_energies = {}
    with open(exp_data_csv, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip the header
        for row in reader:
            compound, energy = row[0], float(row[1])  # Assuming energy is a float
            exp_energies[compound] = energy

    headers = ['Compound Name', 'Empirical Formula', 'U Parameter (eV)', 'Total Energy From Vasp (eV)', 
               'Energy Per Sulfur (eV)', 'Comp Formation Energy', 'Experimental Formation Energy (eV/atom)']

    # Open the file in write mode
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        # Iterate over the entries and write the data rows
        for entry in entries:
            compound_name = entry.get_compound_name()  # Adjust this based on your entries structure
            empirical_formula_str, num_units = empirical_formula(compound_name)
            u_parameter = entry.get_u_parameter()  # Adjust this based on your entries structure
            total_energy = entry.total_energy  # Adjust this based on your entries structure

            # Check if num_units is zero to avoid division by zero
            if num_units == 0:
                continue

            # Calculate the formation energy
            formula_elements = parse_formula(empirical_formula_str)
            total_element_energy = sum(elemental_energies.get(elem, 0) * count for elem, count in formula_elements.items())
            energy_per_unit = total_energy / num_units
            formation_energy = energy_per_unit - total_element_energy

            exp_energy = exp_energies.get(compound_name, 'N/A')

            # Write the row to the CSV file
            writer.writerow([compound_name, empirical_formula_str, u_parameter, total_energy, energy_per_unit, formation_energy, exp_energy])

    print(f"CSV file created at {csv_file}")

