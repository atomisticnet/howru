"""
Enumerate all possible/relevant reactions.

"""

import copy
import numpy as np

from pymatgen.analysis.reaction_calculator import Reaction, ReactionError

# from .util import powerset
from util import powerset


__date__ = "2020-07-23"
__version__ = "0.1"


class Reactions(object):

    def __init__(self, reactants, products, required_compounds=None,
                 all_compounds=None, metal_correction_species=None):
        """
        Enumerate all reactions of any combination of the reactants to any
        combination of products.  Make sure that the required compounds
        are involved in the reactions.

        """

        self.reactants = [
            c for c in reactants if not np.isnan(c.expt_formation_energy_RT)]
        self.products = [
            c for c in products if not np.isnan(c.expt_formation_energy_RT)]
        if all_compounds is None:
            self.all_compounds = list(set(self.reactants) | set(self.products))
        else:
            self.all_compounds = all_compounds
        if not set(self.reactants).issubset(self.all_compounds):
            raise ValueError("(Some of) the reactants are not in the "
                             "list of 'all compounds'.")
        if not set(self.products).issubset(self.all_compounds):
            raise ValueError("(Some of) the products are not in the "
                             "list of 'all compounds'.")
        if required_compounds is not None:
            self.required_compounds = [
                c for c in required_compounds
                if not np.isnan(c.expt_formation_energy_RT)]
            if (required_compounds is not None) and (
                    not set(required_compounds).issubset(self.all_compounds)):
                raise ValueError("(Some of) the required compounds are "
                                 "neither reactants nor products.")
        else:
            self.required_compounds = None

        self.formation_enthalpy_expt = np.array([
            c.expt_formation_energy_RT for c in self.all_compounds])

        # enumerate possible reactions and calculate experimental
        # reaction energies
        self.reactions = []
        self.reaction_energies_expt = []
        self.reaction_matrix = []
        self.num_atoms = []
        self._enumerate_reactions()

        # if metal corrections following Jain are used, determine the
        # number of transition-metal (TM) species in all oxide compounds
        self.metal_correction_species = metal_correction_species
        if metal_correction_species is None:
            self.num_TM = None
        else:
            num_TM = []
            for c in self.all_compounds:
                if "O" not in c.composition:
                    TM_counts = [0 for M in metal_correction_species]
                else:
                    TM_counts = []
                    for M in metal_correction_species:
                        TM_counts.append(c.composition[M])
                num_TM.append(TM_counts)
            self.num_TM = np.array(num_TM)

    @property
    def all_compositions(self):
        return [c.composition for c in self.all_compounds]

    def composition_id(self, c):
        if c in self.all_compositions:
            return self.all_compositions.index(c)
        else:
            return None

    def _enumerate_reactions(self):
        """
        Enumerate possible reactions.

        Sets:
          self.reactions

        """
        allowed_prod_comp = [c.composition for c in self.products]
        allowed_reac_comp = [c.composition for c in self.reactants]
        self.reactions = []
        self.reaction_energies_expt = []
        self.reaction_matrix = []

        def num_atoms(reaction):
            return round(np.sum([0.5*abs(c)*reaction.all_comp[i].num_atoms
                                 for i, c in enumerate(reaction.coeffs)]))

        for product in self.products:
            reactants = []
            for b in self.reactants:
                if set(b.composition.elements).issubset(
                        set(product.composition.elements)) and b != product:
                    reactants.append(b)
            for idx in powerset(range(len(reactants))):
                reactant_combo = [reactants[i].composition for i in idx]
                try:
                    r = Reaction(reactant_combo, [product.composition])
                except ReactionError:
                    continue
                if len(r.products) > 1:
                    continue
                # normalize reaction to the product
                r.normalize_to(r.products[0])
                # make sure that all reactants and products are allowed
                if any([c not in allowed_prod_comp for c in r.products]):
                    continue
                if any([c not in allowed_reac_comp for c in r.reactants]):
                    continue
                # make sure that the reaction involves all required
                # compositions
                if self.required_compounds is not None:
                    if any([s.composition not in r.all_comp
                            for s in self.required_compounds]):
                        continue
                    if any([abs(r.get_coeff(s.composition)) < 0.0001
                            for s in self.required_compounds]):
                        continue
                # make sure not to consider redundant reactions
                r_inv = r.copy()
                r_inv._coeffs = -r_inv._coeffs
                if (r in self.reactions) or (r_inv in self.reactions):
                    continue
                r_vec = self.r2vec(r)
                Er = r_vec.dot(self.formation_enthalpy_expt)
                self.reactions.append(r)
                self.reaction_energies_expt.append(Er)
                self.reaction_matrix.append(r_vec)
                self.num_atoms.append(num_atoms(r))
        self.reaction_energies_expt = np.array(self.reaction_energies_expt)
        self.reaction_matrix = np.array(self.reaction_matrix)

    def remove_compound(self, comp):
        """
        Remove a single compound and all reactions that it participates in.
        This method returns a new instance of Reactions.

        """

        out = copy.deepcopy(self)

        idx = [i for i, r in enumerate(self.reactions) if
               comp.composition not in r.all_comp]

        out.reactants = [comp2 for comp2 in self.reactants if comp2 != comp]
        out.products = [comp2 for comp2 in self.products if comp2 != comp]
        if self.required_compounds is not None:
            out.required_compounds = [comp2 for comp2
                                      in self.required_compounds
                                      if comp2 != comp]
        if (out.required_compounds is not None and len(
                out.required_compounds) == 0):
            out.required_compounds = None

        out.reactions = [self.reactions[i] for i in idx]
        out.reaction_energies_expt = np.array([self.reaction_energies_expt[i]
                                               for i in idx])
        out.reaction_matrix = np.array([self.reaction_matrix[i]
                                        for i in idx])
        out.num_atoms = [self.num_atoms[i] for i in idx]

        return out

    def require_compound(self, comp):
        """
        Remove reactions that do not a specific required compound.
        This method returns a new instance of Reactions.

        """

        if comp not in self.all_compounds:
            raise ValueError("Required compound not in composition space.")

        out = copy.deepcopy(self)

        idx = [i for i, r in enumerate(self.reactions) if
               comp.composition in r.all_comp]

        if self.required_compounds is None:
            out.required_compounds = [comp]
        else:
            out.required_compounds = list(
                set(self.required_compounds + [comp]))

        out.reactions = [self.reactions[i] for i in idx]
        out.reaction_energies_expt = np.array([self.reaction_energies_expt[i]
                                               for i in idx])
        out.reaction_matrix = np.array([self.reaction_matrix[i]
                                        for i in idx])
        out.num_atoms = [self.num_atoms[i] for i in idx]

        return out

    def r2vec(self, r):
        """
        Convert a reaction to a vector representation in which each
        component corresponds to the coefficient of a compound.

        Args:
          r: instance of Reaction

        Returns:
          vec: [c_1, c_2, ..., c_N] where c_i is the coefficient of compound
            i. c_i is positive for products and negative for reactants.

        """
        if not set(r.all_comp).issubset(self.all_compositions):
            raise ValueError("Not all reactants and.or products are in the "
                             "present composition space.")
        v = []
        for c in self.all_compositions:
            if c in r.all_comp:
                v.append(r.get_coeff(c))
            else:
                v.append(0.0)
        return np.array(v)

    def energy_comp(self, U, Jain=None):
        """
        Return computed enthalpies for all compounds for a given set of U
        values.

        """
        energies = np.array([c.energy_for_U(U) for c in self.all_compounds])
        if Jain is not None and self.num_TM is not None:
            energies -= self.num_TM.dot(Jain)
        elif self.num_TM is not None:
            raise ValueError("No Jain metal corrections given but "
                             "required for reactions.")
        elif Jain is not None:
            raise ValueError("Jain metal corrections given but species "
                             "not initialized.")
        return energies

    def reaction_energies_comp(self, U, Jain=None):
        """
        Return the computed reaction energies for all reactions for a given
        set of U values.

        """
        if len(self.reactions) == 0:
            return np.array([])
        else:
            return self.reaction_matrix.dot(self.energy_comp(U, Jain=Jain))

    def reaction_energy_errors(self, U, Jain=None):
        """
        Difference of predicted and experimental reaction energies for a
        given set of U values, normalized by the number of atoms.

        """
        if len(self.reactions) > 0:
            # errors = (self.reaction_energies_comp(U, Jain=Jain)
            #           - self.reaction_energies_expt)/self.num_atoms
            errors = (self.reaction_energies_comp(U, Jain=Jain)
                      - self.reaction_energies_expt)
        else:
            errors = np.array([])
        return errors

    def reaction_energy_rmse_mae(self, U, weighted=False, Jain=None):
        errors = self.reaction_energy_errors(U, Jain=Jain)
        if len(errors) == 0:
            mae, rmse = 0.0, 0.0
        else:
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors*errors))
        w = len(errors) if weighted else 1
        return np.array([w*rmse, w*mae])

    def print_reactions(self, U, fname, Jain=None):
        dEr_comp = self.reaction_energies_comp(U, Jain=Jain)
        dEr_expt = self.reaction_energies_expt
        with open(fname, "w") as fp:
            fp.write('{:4s} {:7s} {:7s} {:2s} {}\n'.format(
                "#", "Comput.", "Expt.", "N", "Reaction"))
            for i, r in enumerate(self.reactions):
                fp.write('{:4d} {:7.3f} {:7.3f} {:2d} "{}"\n'.format(
                    i, dEr_comp[i], dEr_expt[i],
                    self.num_atoms[i], str(r)))

    def print_formation_energies(self, U, fname, Jain=None):
        dEr_comp = self.reaction_energies_comp(U, Jain=Jain)
        dEr_expt = self.reaction_energies_expt
        with open(fname, "w") as fp:
            fp.write(('{:4s} {:7s} {:7s} {:2s} {:2s} {:4s} {:2s} {:4s} '
                      '{} {}\n').format(
                          "#", "Comput.", "Expt.", "N", "M1", "Oxi1",
                          "M2", "Oxi2", "Compound", "Reaction"))
            for i, r in enumerate(self.reactions):
                product = r.products[0]
                oxi = ["{:2s} {:4.1f}".format(
                    s.symbol, product.oxi_state_guesses()[0][s.symbol])
                       for s in product if s.is_metal or s.is_metalloid]
                if len(oxi) == 1:
                    oxi.append("{:2s} {:4s}".format("*", "*"))
                fp.write(
                    '{:4d} {:7.3f} {:7.3f} {:2d} {} "{}" "{}"\n'.format(
                        i, dEr_comp[i], dEr_expt[i], int(self.num_atoms[i]),
                        " ".join(oxi), product.reduced_formula, str(r)))
