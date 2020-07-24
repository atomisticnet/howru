"""
Compounds are 'pymatgen.Composition's with additional properties,
most importantly, the compound energy as function of U values.

"""

import numpy as np
import pymatgen as mg
import pandas as pd

import functools
from scipy.optimize import minimize

__date__ = "2020-07-23"
__version__ = "0.1"


class Compound(object):
    def __init__(self, formula, data, TM_species):
        TM_species = np.array(TM_species)
        self.composition, nFU = mg.Composition(
            formula).get_reduced_composition_and_factor()
        idx = ['U({})'.format(s) for s in TM_species]
        U_values = data[idx]
        self.U_dependencies = np.invert(np.isnan(U_values.values[0]))
        idx = np.invert(np.isnan(data['SCAN+rvv10+U Total E (eV)'].values))
        self.U_values = U_values.values[:, self.U_dependencies][idx]
        self.U_species = TM_species[self.U_dependencies]
        self.num_U = len(self.U_species)
        self.SCAN_energies = data['SCAN+rvv10+U Total E (eV)'].values[idx]/nFU
        self.expt_formation_energy_RT = data[
            'Expt. Formation Energy 298.15K (eV)'].values[0]/nFU
        try:
            self.expt_formation_energy_0K = data[
                'Expt. Formation Energy 0K (eV)'].values[0]/nFU
        except KeyError:
            self.expt_formation_energy_0K = self.expt_formation_energy_RT
        if self.num_U == 0:
            self._interpolate_energy = lambda x: [self.SCAN_energies[0]]
        elif self.num_U == 1:
            self.U_values = self.U_values[:, 0].T
            coeff = np.polyfit(self.U_values, self.SCAN_energies, 1)
            self._interpolate_energy = np.poly1d(coeff)
        elif self.num_U == 2:
            UUE = list(zip(self.U_values[:, 0], self.U_values[:, 1],
                           self.SCAN_energies))

            def plane(U1, U2, coeff):
                return coeff[0]*U1 + coeff[1]*U2 + coeff[2]

            def errors(coeff):
                return np.sum([(plane(U1, U2, coeff) - E)**2
                               for U1, U2, E in UUE])

            coeff0 = [0.1, 0.1, 0.1]
            results = minimize(errors, coeff0)
            if not results.success:
                raise ValueError("Energy(U1, U2) fit not converged.")
            self._interpolate_energy = functools.partial(
                plane, coeff=results.x)
        else:
            raise NotImplementedError(
                "Dependency on more than 2 U values not implemented.")

    def __str__(self):
        out = 'Composition: {}'.format(self.composition.formula)
        out += '\nFormation energy RT: {} eV'.format(
            self.expt_formation_energy_RT)
        out += '\nFormation energy 0K: {} eV'.format(
            self.expt_formation_energy_0K)
        out += '\nU species: ' + str(self.U_species)
        out += '\nU Values: ' + str(self.U_values)
        out += '\nEnergies: ' + str(self.SCAN_energies)
        return out

    def __repr__(self):
        return self.composition.__repr__()

    def energy_for_U(self, U):
        U_values = np.array(U)[self.U_dependencies]
        if len(U_values) > 1:
            E = self._interpolate_energy(*U_values)
        else:
            E = self._interpolate_energy(U_values)[0]
        return E


def read_compounds_from_csv(csv_file, TM_species):
    """
    Parse a comma separated value file with compound information.

    """
    data = pd.read_csv(csv_file, index_col=None)
    comp = list(set(data["Composition"]))
    comp = sorted([c for c in comp if isinstance(c, str)])
    compounds = []
    for c in comp:
        idx = data["Composition"] == c
        compounds.append(Compound(c, data[idx], TM_species))
    return compounds


def read_elements_from_csv(csv_file, TM_species):
    """
    Parse CSV file with element information.

    """
    data = pd.read_csv(csv_file)
    comp = list(set(data["Composition"]))
    comp = sorted([c for c in comp if isinstance(c, str)])
    U = {'U(Sc)': [float("nan")], 'U(Ti)': [float("nan")],
         'U(V)': [float("nan")], 'U(Cr)': [float("nan")],
         'U(Mn)': [float("nan")], 'U(Fe)': [float("nan")],
         'U(Co)': [float("nan")], 'U(Ni)': [float("nan")],
         'U(Cu)': [float("nan")], 'U(Zn)': [float("nan")]}
    compounds = []
    for c in comp:
        idx = data["Composition"] == c
        d = U.copy()
        d['SCAN+rvv10+U Total E (eV)'] = [
            data[idx]['SCAN+rvv10+U Total E (eV)'].values[-1]]
        d['Expt. Formation Energy 298.15K (eV)'] = [0.0]
        d['Composition'] = [c]
        compounds.append(Compound(c, pd.DataFrame(d), TM_species))
    return compounds
