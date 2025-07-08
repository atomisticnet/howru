"""
Collection of utility functions.

"""

import itertools
import pymatgen
from pymatgen.core.composition import Composition
# idk what this shoujld be self.composition = Composition(...) 
from pymatgen.apps.borg.hive import VaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen
import csv
import re
import math
from math import gcd
from functools import reduce
import pandas as pd

__date__ = "2024-01-17"
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


def parse_formula(formula):
    """
    Parse the chemical formula into elements and their counts.
    
    Arguments:

    formula (str) chemical formula that will be parsed
    """
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    return {elem: int(count) if count else 1 for elem, count in elements}


def calculate_gcd(numbers):
    if not numbers:
        return 1
    integers = [int(num * 1e10) for num in numbers]
    result = integers[0]
    for num in integers[1:]:
        result = math.gcd(result, num)
    return result / 1e10


def get_u_parameters(entries, preferred_elements=['Fe','Cu','V','Ti','Cr','Mn','Co','Ni']):
    """
    Extract the U parameters for preferred_elements found in DFT calculations

    Arguments:
    entries (list): List of ComputedEntry objects.
    preferred_elements (list): List of element symbols in order of preference.

    Returns:
    dict: A dictionary containing U parameters for preferred elements in each entry.
    """
    u_parameters = {}
    
    for entry in entries:
        hubbards = entry.parameters.get('hubbards', {})
        u_values = {}
        for element in preferred_elements:
            u_values[element] = hubbards.get(element)
        u_parameters[entry.composition.reduced_formula] = u_values

    return u_parameters


def empirical_formula(compound_dict):
    """
    Calculate the empirical formula and the number of atoms in a single 
    formula unit formula units from a composition dictionary.

    Arguments:
    compound_dict (dict)    The composition of the compound, 
                            where keys are element symbols and values are 
                            their counts.
    Returns: 
    empirical_formula (str) Returns the composition of each individual 
                            compound 
    num_atoms (int)         Number of atoms in the empirical formula
    num_formula (int)       Number of formula units for a given compound
    """
    elements_counts = compound_dict
    counts = list(elements_counts.values())

    if all(count == 0 for count in counts):
        return "", 0 

    gcd_counts = calculate_gcd(counts) if counts else 1 

    simplest_ratio = {element: count // gcd_counts for element, count in 
                      elements_counts.items() if gcd_counts != 0}

    empirical_formula = ''.join(f"{element}{(count if count > 1 else '')}" 
                                for element, count in simplest_ratio.items())

    num_atoms = sum(simplest_ratio.values())
    
    num_formula = gcd_counts

    num_sulfur = simplest_ratio.get('S',0)
    
    return empirical_formula, num_atoms, num_formula, num_sulfur



def get_compound_name_from_dict(compound_dict):
    """
    Generates a compound name from a composition dictionary.
    
    Arguments:
    compound_dict (dict)  The composition of the compound, where keys are 
                          element symbols and values are their counts.
                          
    Returns:
    compound_name (str)   A string representing the compound name.
    """
    sorted_items = sorted(compound_dict.items())
    compound_name = ''.join(f"{element}{int(count) if count > 1 else ''}"
                            for element, count in sorted_items)
    return compound_name


def load_elemental_energies(csv_filepath):
    """
    Load elemental energies from a CSV file into a dictionary.

    Arguments:
    csv_filepath (str)        Path to the CSV file containing elemental 
                              energies.

    Returns:
    elemental_energies (dict) A dictionary with element symbols as keys and 
                              their energies as values.
    """
    elemental_energies = {}
    with open(csv_filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) 
        for row in reader:
            element, energy = row[0], float(row[1])
            elemental_energies[element] = energy
    return elemental_energies


csv_filepath = 'BCD4 - Calculated Energies.csv'
elemental_energies = load_elemental_energies(csv_filepath)


def load_experimental_data(csv_filepath):
    """
    Load experimental formation energies from a CSV file into a dictionary.

    Arguments:
    csv_filepath (str)        Path to the CSV file containing experimental energies.

    Returns:
    experimental_energies (dict) A dictionary with compound names as keys and 
                                  their experimental energies as values.
    """
    experimental_energies = {}
    with open(csv_filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip the header
        for row in reader:
            compound, energy = row[0], float(row[1])
            experimental_energies[compound] = energy
    print(experimental_energies)  # Print the dictionary to check its contents
    return experimental_energies


exp_data_csv = 'Experimental Energy - ExpEnergy.csv'
experimental_energies = load_experimental_data(exp_data_csv)

def verify_elemental_energies(elemental_energies):
    expected_elements = ['Fe', 'Cu', 'Ti', 'Cr', 'Mn', 'Co', 'Ni']

    if not elemental_energies:
        print("Elemental energies dictionary is empty.")
        return False

    for element in expected_elements:
        if element not in elemental_energies:
            print(f"Element {element} is missing from the elemental energies dictionary.")
            return False

    print("Elemental energies dictionary is verified.")
    return True


def collect_elemental_energies(entries):
    """
    Extracts and returns elemental energies from VASP entries.

    Arguments:
    entries (list): List of ComputedEntry objects.

    Returns:
    dict: A dictionary with element symbols as keys and their energies as values.
    """
    elemental_energies = {}
    for entry in entries:
        composition_dict = entry.composition.as_dict()
        total_energy = entry.energy
        if len(composition_dict) == 1:  # Ensure it's an elemental compound
            element = list(composition_dict.keys())[0]
            energy_per_atom = total_energy / composition_dict[element]
            elemental_energies[element] = energy_per_atom
            print(f"Element: {element}, Energy per Atom: {energy_per_atom}")  # Debug print
    print(f"Collected elemental energies: {elemental_energies}")  # Debug print
    return elemental_energies

def save_elemental_energies_to_csv(elemental_energies, filepath):
    """
    Saves elemental energies to a CSV file.

    Arguments:
    elemental_energies (dict): Dictionary with element symbols as keys and their energies as values.
    filepath (str): Path to save the CSV file.
    """
    df = pd.DataFrame(list(elemental_energies.items()), columns=['Element', 'SCAN+rvv10+U Total E (eV)'])
    print(f"DataFrame to save: {df}")  # Debug print
    df.to_csv(filepath, index=False)
    print(f"Elemental energies saved to {filepath}")

def create_csv_from_entries(entries, csv_file, elemental_energies, experimental_energies):
    """
    Create a CSV file from VASP calculation entries.
    
    Arguments:
    entries (list)      List of entries from load_entries_json function
    csv_file (str)      Path to save the CSV file
    exp_data_csv (str)  Path to the CSV file with experimental 
                        formation energies
    Returns:
    output_data (csv)   csv file 
    """
    if not entries:
        print("No entries to create a CSV from.")
        return

    # Load experimental data
    experimental_energies = load_experimental_data(exp_data_csv)

    preferred_elements = ['Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
    headers = ['Composition'] + [f'U {el}' for el in preferred_elements] + \
          ['SCAN+rvv10+U_Total_E_(eV)', 'Energy_per_S_(eV)', 'Calc_Formation_Energy_(eV)', 'Expt_Formation_Energy_298.15K_(eV)']


    rows = []

    for entry in entries:
        composition_dict = entry.composition.as_dict()
        total_energy = entry.energy

        empirical_formula_str, num_atoms, num_formula, num_sulfur = empirical_formula(composition_dict)

        if num_atoms == 0 or len(composition_dict) == 1:
            print(f"Skipping single-element entry: {entry.composition.reduced_formula}")
            continue

        if len(composition_dict) == 1:
            energy_per_formula_unit = total_energy / num_atoms
        else:
            energy_per_formula_unit = total_energy / num_formula

        u_values_dict = get_u_parameters([entry], preferred_elements)
        
        total_element_energy = sum(elemental_energies.get(elem, 0) * count for elem, count in composition_dict.items())
        element_energy_per_formula_unit = total_element_energy / num_formula
        
        formation_energy_per_formula_unit = energy_per_formula_unit - element_energy_per_formula_unit
        formation_energy_per_atom = formation_energy_per_formula_unit / num_atoms

        compound_name = entry.composition.reduced_formula
        exp_energy = experimental_energies.get(compound_name, 'N/A')

        row_data = [compound_name]

        for el in preferred_elements:
            u_value = u_values_dict.get(compound_name, {}).get(el, '')
            row_data.append(u_value)

        energy_per_sulfur = element_energy_per_formula_unit / num_sulfur if num_sulfur > 0 else 0

        row_data += [total_energy, energy_per_sulfur, formation_energy_per_atom, exp_energy]
        rows.append(row_data)

    # Sort rows by compound name
    rows.sort(key=lambda x: x[0])

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"CSV file created at {csv_file}")



    print(f"CSV file created at {csv_file}")
    for entry in entries:
        print("Composition Dict:", entry.composition.as_dict())
        print("Total Energy:", entry.energy)
        composition_dict = entry.composition.as_dict()
        empirical_formula_str, num_atoms, num_formula, num_sulfur = empirical_formula(composition_dict)
        print("Empirical Formula:", empirical_formula_str)
        print("Number of Atoms:", num_atoms)
        print("Number of Formula Units:", num_formula)
        print("Number of Sulfur:", num_sulfur)
        
        u_values_dict = get_u_parameters([entry], preferred_elements)
        print("U Values Dict:", u_values_dict)
        total_element_energy = sum(elemental_energies.get(elem, 0) *
                               count for elem, count in composition_dict.items())
        print("Total Element Energy:", total_element_energy)
        element_energy_per_formula_unit = total_element_energy / num_formula
        print("Element Energy per Unit:", element_energy_per_formula_unit)
        if len(composition_dict) == 1:
                energy_per_formula_unit = total_energy / num_atoms
                print("DFT Energy per Formula Unit:", energy_per_formula_unit)
        else:
                energy_per_formula_unit = total_energy / num_formula
                formation_energy_per_formula_unit = energy_per_formula_unit - element_energy_per_formula_unit
                print("DFT Energy per Formula Unit:", energy_per_formula_unit)
                print("Formation Energy per Unit:", formation_energy_per_formula_unit) 
        print()