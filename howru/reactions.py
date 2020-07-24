"""
Enumerate all possible/relevant reactions.

"""

import math
import numpy as np

import pymatgen as mg
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError

from .util import powerset

__date__ = "2020-07-23"
__version__ = "0.1"


class Reactions(object):

    def __init__(self, elements, atoms, dimers, binary_oxides,
                 ternary_oxides):

        self.elements = elements
        self.atoms = atoms
        self.dimers = dimers
        self.binary_oxides = binary_oxides
        self.ternary_oxides = ternary_oxides

        self.binary_oxide_reactions = None
        self.ternary_oxide_reactions = None
        self.binary_o2_reactions = None
        self.ternary_o2_reactions = None
        self.binary_oxide_formations = None
        self.ternary_oxide_Formations = None
        self.dimer_binding_energies = None

    def __str__(self):
        return

    def num_atoms(self, reaction):
        """ Number of atoms involved in a reaction. """
        return round(np.sum([0.5*abs(c)*reaction.all_comp[i].num_atoms
                             for i, c in enumerate(reaction.coeffs)]))

    def energy_correction_term(self, reaction, metal_corrections):
        """
        Evaluate the correction term of a reaction energy.

        Arguments:
          reaction (Reaction): the reaction
          metal_corrections (dict): correction term for each metal species

        Returns:
          energy (float)

        """
        corr_energy = 0.0
        for i, comp in enumerate(reaction.all_comp):
            if "O" not in comp:
                continue
            for M in metal_corrections:
                if M in comp:
                    corr_energy -= reaction.coeffs[
                        i]*comp[M]*metal_corrections[M]
        return corr_energy

    def get_reactions(self, allowed_reactants, allowed_products, U,
                      required_compounds=None):
        """
        Enumerate possible reactions.

        Arguments:
          allowed_reactants ([Compound]): List of allowed reactant compounds
          allowed_products ([Compound]): List of allowed product compounds
          U (list): List of U values
          require_compounds ([Compound]): Compounds that are required to
            occur in each reaction

        Returns:
          reactions (dict): Key are pymatgen Reactions (r) and values are
            are the computed (dEr_comp) and experimental (dEr_expt) reaction
            energies: {r: [dEr_comp, dEr_expt]}

        """
        allowed_prod_comp = [c.composition for c in allowed_products]
        allowed_reac_comp = [c.composition for c in allowed_reactants]
        reactions = {}
        for product in allowed_products:
            reactants = []
            for b in allowed_reactants:
                if set(b.composition.elements).issubset(
                        set(product.composition.elements)) and b != product:
                    reactants.append(b)
            for idx in powerset(range(len(reactants))):
                reactant_combo = [reactants[i].composition for i in idx]
                try:
                    r = Reaction(reactant_combo, [product.composition])
                except ReactionError:
                    continue
                # make sure that all reactants and products are allowed
                if any([c not in allowed_prod_comp for c in r.products]):
                    continue
                if any([c not in allowed_reac_comp for c in r.reactants]):
                    continue
                # make sure that the reaction involves all required
                # compositions
                if required_compounds is not None:
                    if any([s.composition not in r.all_comp
                            for s in required_compounds]):
                        continue
                    if any([abs(r.get_coeff(s.composition)) < 0.0001
                            for s in required_compounds]):
                        continue
                try:
                    dEr_expt = 0.0
                    dEr_comp = 0.0
                    for i, ir in enumerate(idx):
                        E_reactant_expt = reactants[
                            ir].expt_formation_energy_RT
                        E_reactant_comp = reactants[ir].energy_for_U(U)
                        dEr_expt += r.coeffs[i]*E_reactant_expt
                        dEr_comp += r.coeffs[i]*E_reactant_comp
                    dEr_expt += r.coeffs[-1]*product.expt_formation_energy_RT
                    dEr_comp += r.coeffs[-1]*product.energy_for_U(U)
                except ValueError:
                    # some energies are not available for requested U values
                    continue
                # make sure that the experimental energies are available
                if math.isnan(dEr_expt):
                    continue
                # finally, make sure not to consider redundant reactions
                r_inv = r.copy()
                r_inv._coeffs = -r_inv._coeffs
                if (r not in reactions) and (r_inv not in reactions):
                    reactions[r] = [dEr_comp, dEr_expt]
        return reactions

    def eval_binary_oxide_reactions(self, U, fname=None):
        """
        Evaluate all oxide reactions with binary TM oxides as product (not
        mixed metal oxides) that are not involving elemental oxygen.  If a
        file name is given, the reaction energies are written to an output
        file.

        Arguments:
          U ([float]): U values for all 3d TMs
          fname (str): name of the output file
          weighted (bool): if True, multiply error by number of reactions

        """
        allowed_products = self.binary_oxide
        allowed_reactants = self.binary_oxide + self.ternary_oxide
        reactions = self.get_reactions(allowed_reactants,
                                       allowed_products, U)
        if fname is not None:
            with open(fname, "w") as fp:
                fp.write('{:4s} {:7s} {:7s} {:2s} {}\n'.format(
                    "#", "Comput.", "Expt.", "N", "Reaction"))
                for i, r in enumerate(reactions):
                    dEr_comp, dEr_expt = reactions[r]
                    fp.write('{:4d} {:7.3f} {:7.3f} {:2d} "{}"\n'.format(
                        i, dEr_comp, dEr_expt, int(self.num_atoms(r)), str(r)))
        self.binary_oxide_reactions = reactions

    def eval_ternary_oxide_reactions(self, U, fname=None):
        """
        Evaluate all oxide reactions with mixed TM oxides as product that
        are not involving elemental oxygen.  If a file name is given,
        the reaction energies are written to an output file.

        Arguments:
          U ([float]): U values for all 3d TMs
          fname (str): name of the output file

        """
        allowed_products = self.ternary_oxides
        # allowed_reactants = self.binary_oxides + self.ternary_oxides +
        # mixed_comp
        allowed_reactants = self.binary_oxides + self.ternary_oxides
        reactions = self.get_reactions(allowed_reactants, allowed_products, U)
        if fname is not None:
            with open(fname, "w") as fp:
                fp.write('{:4s} {:7s} {:7s} {:2s} {}\n'.format(
                    "#", "Comput.", "Expt.", "N", "Reaction"))
                for i, r in enumerate(reactions):
                    dEr_comp, dEr_expt = reactions[r]
                    fp.write('{:4d} {:7.3f} {:7.3f} {:2d} "{}"\n'.format(
                        i, dEr_comp, dEr_expt, int(self.num_atoms(r)), str(r)))
        self.ternary_oxide_reactions = reactions

    def eval_binary_o2_reactions(self, U, fname=None):
        """
        Evaluate all reactions involving elemental oxygen with binary oxide
        products.  If a file name is given, the reaction energies are
        written to an output file.

        Arguments:
          U ([float]): U values for all 3d TMs
          fname (str): name of the output file

        """
        O2 = [c for c in self.elements
              if c.composition == mg.Composition("O2")]
        allowed_products = self.binary_oxides
        allowed_reactants = self.binary_oxides + self.ternary_oxides + O2
        reactions = self.get_reactions(allowed_reactants,
                                       allowed_products, U,
                                       required_compounds=O2)
        if fname is not None:
            with open(fname, "w") as fp:
                fp.write('{:4s} {:7s} {:7s} {:2s} {}\n'.format(
                    "#", "Comput.", "Expt.", "N", "Reaction"))
                for i, r in enumerate(reactions):
                    dEr_comp, dEr_expt = reactions[r]
                    fp.write('{:4d} {:7.3f} {:7.3f} {:2d} "{}"\n'.format(
                        i, dEr_comp, dEr_expt, int(self.num_atoms(r)), str(r)))
        self.binary_oxide_reactions = reactions

    def eval_ternary_o2_reactions(self, U, fname=None):
        """
        Evaluate all reactions involving elemental oxygen with mixed metal
        oxide products.  If a file name is given, the reaction energies
        are written to an output file.

        Arguments:
          U ([float]): U values for all 3d TMs
          fname (str): name of the output file

        """
        O2 = [c for c in self.elements
              if c.composition == mg.Composition("O2")]
        allowed_products = self.ternary_oxides
        allowed_reactants = self.binary_oxidess + self.ternary_oxides + O2
        reactions = self.get_reactions(allowed_reactants,
                                       allowed_products, U,
                                       required_compounds=O2)
        if fname is not None:
            with open(fname, "w") as fp:
                fp.write('{:4s} {:7s} {:7s} {:2s} {}\n'.format(
                    "#", "Comput.", "Expt.", "N", "Reaction"))
                for i, r in enumerate(reactions):
                    dEr_comp, dEr_expt = reactions[r]
                    fp.write('{:4d} {:7.3f} {:7.3f} {:2d} "{}"\n'.format(
                        i, dEr_comp, dEr_expt, int(self.num_atoms(r)), str(r)))
        self.ternary_o2_reactions = reactions

    def eval_formation_energies(self, oxides, U, elements=None,
                                metal_corrections=None, fname=None,
                                weighted=False):
        """
        Evaluate oxide formation energies.  If a file name is given, the
        formation energies are written to an output file.

        Arguments:
          oxides ([Compound]): list of oxide compounds
          U ([float]): U values for all 3d TMs
          elements ([Compound]): list of elemental compounds
          fname (str): name of the output file
          weighted (bool): if True, multiply error by number of reactions

        Returns:
          (rmse, mae): errors normalized per atom unless return_all is True
          (composition, expt, comput) arrays if return_all is True

        """
        allowed_products = oxides
        if elements is None:
            allowed_reactants = self.elements
        else:
            allowed_reactants = elements
        reactions = self.get_reactions(allowed_reactants,
                                       allowed_products, U)
        compositions = []
        Ef_calc = []
        Ef_expt = []
        if fname is not None:
            with open(fname, "w") as fp:
                fp.write(('{:4s} {:7s} {:7s} {:2s} {:2s} {:4s} {:2s} {:4s} '
                          '{} {}\n').format(
                              "#", "Comput.", "Expt.", "N", "M1", "Oxi1",
                              "M2", "Oxi2", "Compound", "Reaction"))
                for i, r in enumerate(reactions):
                    if metal_corrections is not None:
                        reactions[r][0] += self.energy_correction_term(
                            r, metal_corrections)
                    dEr_comp, dEr_expt = reactions[r]
                    product = r.products[0]
                    oxi = ["{:2s} {:4.1f}".format(
                        s.symbol, product.oxi_state_guesses()[0][s.symbol])
                           for s in product if s.is_metal or s.is_metalloid]
                    if len(oxi) == 1:
                        oxi.append("{:2s} {:4s}".format("*", "*"))
                    fp.write(
                        '{:4d} {:7.3f} {:7.3f} {:2d} {} "{}" "{}"\n'.format(
                            i, dEr_comp, dEr_expt, int(self.num_atoms(r)),
                            " ".join(oxi), product.reduced_formula, str(r)))
                    compositions.append(product)
                    Ef_calc.append(dEr_comp)
                    Ef_expt.append(dEr_expt)
        else:
            for i, r in enumerate(reactions):
                if metal_corrections is not None:
                    reactions[r][0] += self.energy_correction_term(
                        r, metal_corrections)
                dEr_comp, dEr_expt = reactions[r]
                product = r.products[0]
                compositions.append(product)
                Ef_calc.append(dEr_comp)
                Ef_expt.append(dEr_expt)
        return reactions, (compositions, np.array(Ef_expt), np.array(Ef_calc))

    def eval_binary_oxide_formations(self, U, metal_corrections=None,
                                     fname=None):
        self.binary_oxide_formations, _ = self.eval_formation_energies(
            self.binary_oxides, U, metal_corrections=metal_corrections,
            fname=fname)

    def eval_ternary_oxide_formations(self, U, metal_corrections=None,
                                      fname=None):
        self.ternary_oxide_formations, _ = self.eval_formation_energies(
            self.ternary_oxides, U, metal_corrections=metal_corrections,
            fname=fname)

    def eval_dimer_binding_energies(self, U, metal_corrections=None,
                                    fname=None):
        self.dimer_binding_energies, _ = self.eval_formation_energies(
            self.dimers, U, elements=self.atoms,
            metal_corrections=metal_corrections, fname=fname)

    def eval_reaction_errors(self, weighted=False):
        """
        Evaluate the reaction energy errors of all currently stored
        reactions.

        """
        if self.binary_oxide_reactions is not None:
            w = len(self.binary_oxide_reactions) if weighted else 1
            self.binary_oxide_reaction_err = self.calc_errors(
                self.binary_oxide_reactions)*w
        if self.ternary_oxide_reactions is not None:
            w = len(self.ternary_oxide_reactions) if weighted else 1
            self.ternary_oxide_reaction_err = self.calc_errors(
                self.ternary_oxide_reactions)*w
        if self.binary_o2_reactions is not None:
            w = len(self.binary_o2_reactions) if weighted else 1
            self.binary_o2_reaction_err = self.calc_errors(
                self.binary_o2_reactions)*w
        if self.ternary_o2_reactions is not None:
            w = len(self.ternary_o2_reactions) if weighted else 1
            self.ternary_o2_reaction_err = self.calc_errors(
                self.ternary_o2_reactions)*w
        if self.binary_oxide_formations is not None:
            w = len(self.binary_oxide_formations) if weighted else 1
            self.binary_oxide_formation_err = self.calc_errors(
                self.binary_oxide_formations)*w
        if self.ternary_oxide_Formations is not None:
            w = len(self.ternary_oxide_formations) if weighted else 1
            self.ternary_oxide_formation_err = self.calc_errors(
                self.ternary_oxide_formations)*w
