"""
Class implementing the U value optimization.

"""

import os
import numpy as np
from scipy.optimize import least_squares
import pymatgen as mg

from .compound import read_compounds_from_csv, read_elements_from_csv
from .reactions import Reactions

__date__ = "2020-07-23"
__version__ = "0.1"


class UOptimizer(object):

    def __init__(self, TM_species, elements_csv, binary_oxides_csv,
                 ternary_oxides_csv, loocv=False):

        def has_E_expt(compounds):
            return [c for c in compounds
                    if not np.isnan(c.expt_formation_energy_RT)]

        self.TM_species = TM_species
        self.elements = has_E_expt(read_elements_from_csv(
            elements_csv, TM_species))
        self.binary_oxides = has_E_expt(read_compounds_from_csv(
            binary_oxides_csv, TM_species))
        self.ternary_oxides = has_E_expt(read_compounds_from_csv(
            ternary_oxides_csv, TM_species))

        self.all_compounds = list(set(self.elements + self.binary_oxides
                                      + self.ternary_oxides))

        self.U = np.zeros(len(self.TM_species))

        self.metal_corrections = None

        self.reactions = self._get_reactions(self.elements,
                                             self.binary_oxides,
                                             self.ternary_oxides)

        self.reaction_weight = None

        self.loocv = loocv
        if loocv:
            self._loocv_reactions_setup()
        else:
            self.reactions_one_out = None
            self.reactions_one_in = None
            self.loocv_comp = None
            self.all_reactions = None

        self.errors = {}

    def _get_reactions(self, elements, binary_oxides, ternary_oxides,
                       required_compounds=None):
        reactions = {}
        reactions['Binary oxide reactions'] = Reactions(
            reactants=(binary_oxides + ternary_oxides),
            products=binary_oxides,
            required_compounds=required_compounds,
            all_compounds=self.all_compounds)
        reactions['Ternary oxide reactions'] = Reactions(
            reactants=(binary_oxides + ternary_oxides),
            products=ternary_oxides,
            required_compounds=required_compounds,
            all_compounds=self.all_compounds)
        reactions['Binary oxide formation energies'] = Reactions(
            reactants=elements,
            products=binary_oxides,
            required_compounds=required_compounds,
            all_compounds=self.all_compounds,
            metal_correction_species=self.TM_species)
        reactions['Ternary oxide formation energies'] = Reactions(
            reactants=elements,
            products=ternary_oxides,
            required_compounds=required_compounds,
            all_compounds=self.all_compounds,
            metal_correction_species=self.TM_species)
        O2 = [c for c in self.elements
              if c.composition == mg.core.Composition("O2")]
        if required_compounds is not None:
            required_compounds = required_compounds + O2
        else:
            required_compounds = O2
        reactions['Binary O2 reactions'] = Reactions(
            reactants=(binary_oxides + ternary_oxides + O2),
            products=binary_oxides,
            required_compounds=required_compounds,
            all_compounds=self.all_compounds)
        reactions['Ternary O2 reactions'] = Reactions(
            reactants=(binary_oxides + ternary_oxides + O2),
            products=ternary_oxides,
            required_compounds=required_compounds,
            all_compounds=self.all_compounds)
        return reactions

    def _loocv_reactions_setup(self):
        self.reactions_one_out = []
        self.reactions_one_in = []
        self.loocv_comp = []
        self.all_reactions = self.reactions
        N = len(self.binary_oxides)
        for i in range(N):
            comp = self.binary_oxides[i]
            one_out = {}
            one_in = {}
            for kind in self.reactions:
                one_out[kind] = (
                    self.reactions[kind].remove_compound(comp))
                one_in[kind] = (
                    self.reactions[kind].require_compound(comp))
            self.loocv_comp.append(comp)
            self.reactions_one_out.append(one_out)
            self.reactions_one_in.append(one_in)
        N = len(self.ternary_oxides)
        for i in range(N):
            comp = self.ternary_oxides[i]
            one_out = {}
            one_in = {}
            for kind in self.reactions:
                one_out[kind] = (
                    self.reactions[kind].remove_compound(comp))
                one_in[kind] = (
                    self.reactions[kind].require_compound(comp))
            self.loocv_comp.append(comp)
            self.reactions_one_out.append(one_out)
            self.reactions_one_in.append(one_in)

    def reaction_matrix(self):
        """
        Return matrix representation of all all reactions.

        """

        reaction_types = [
            'Binary oxide reactions',
            'Ternary oxide reactions',
            'Binary O2 reactions',
            'Ternary O2 reactions',
            'Binary oxide formation energies',
            'Ternary oxide formation energies']

        reaction_matrix = []
        energies_expt = []
        reactions = []
        num_atoms = []

        for kind in reaction_types:
            reaction_matrix.extend(
                self.reactions[kind].reaction_matrix.tolist())
            energies_expt.extend(
                self.reactions[kind].reaction_energies_expt.tolist())
            reactions.extend(self.reactions[kind].reactions)
            num_atoms.extend(self.reactions[kind].num_atoms)

        return (np.array(reaction_matrix),
                np.array(energies_expt), reactions, num_atoms)

    def compound_energies(self, U, Jain=None):
        """
        Return the predicted energies of all compounds.

        """

        if Jain is None:
            Jain = [0.0 for M in self.TM_species]
        elif Jain == 'auto':
            self.fit_metal_corrections(U)
            Jain = [self.metal_corrections[M] for M in self.TM_species]

        # select formation energies so that the Jain correction is
        # available
        compound_energies = (
            self.reactions['Binary oxide formation energies'].energy_comp(
                U, Jain=Jain))

        return compound_energies

    def fit_metal_corrections(self, U):
        """
        Fit the metal correction energy term following Jain et al. for all
        TM species.

        Arguments:
          U (list): list of U values

        """
        r = self.reactions['Binary oxide formation energies']
        formation_reactions = r.reactions
        formation_energies_expt = r.reaction_energies_expt
        formation_energies_calc = r.reaction_energies_comp(
            U, Jain=[0 for u in U])

        metal_corrections = {}
        for M in self.TM_species:
            idx = np.where([mg.core.Composition(M) in r.all_comp
                            for r in formation_reactions])[0]
            if len(idx) == 0:
                metal_corrections[M] = 0.0
            else:
                comp = [formation_reactions[i].products[0] for i in idx]
                Ef_expt = np.array([formation_energies_expt[i] for i in idx])
                Ef_calc = np.array([formation_energies_calc[i] for i in idx])
                num_atoms = np.array([c.num_atoms for c in comp])
                M_content = np.array([c[M] for c in comp])/num_atoms
                errors = (Ef_expt - Ef_calc)/num_atoms
                if len(comp) > 1:
                    M_content = M_content[:, np.newaxis]
                    slope, _, _, _ = np.linalg.lstsq(
                        M_content, errors, rcond=None)
                    metal_corrections[M] = -slope[0]
                else:
                    metal_corrections[M] = -errors[0]/M_content[0]
        self.metal_corrections = metal_corrections

    def print_errors(self):
        for e in self.errors:
            print((e + " (RMSE & MAE)").ljust(45)
                  + " : {:7.4f} {:7.4f}".format(*self.errors[e]))

    def print_U_values(self, U):
        print("      " + (len(self.TM_species)*"{:5s} "
                          ).format(*list(self.TM_species)))
        print(" U = [" + ", ".join(["{:5.2f}".format(uval).strip()
                                    for uval in U]) + "]")

    def print_Jain(self, metal_corrections=None):
        if metal_corrections is not None:
            m = metal_corrections
        else:
            m = self.metal_corrections
        if m is None:
            Jain = [0.0 for M in self.TM_species]
        else:
            Jain = m.values()
        print("      " + (len(self.TM_species)*"{:5s} "
                          ).format(*list(self.TM_species)))
        print(" c = [" + ", ".join(
            ["{:5.2f}".format(c).strip()
             for c in Jain]) + "]")

    def eval_energy_errors(self, U, verbose=False,
                           print_iterations=False,
                           binary_oxide_reactions=True,
                           ternary_oxide_reactions=True,
                           binary_o2_reactions=True,
                           ternary_o2_reactions=True,
                           binary_oxide_formations=True,
                           ternary_oxide_formations=True,
                           fname_binary_oxide_reactions=None,
                           fname_ternary_oxide_reactions=None,
                           fname_binary_o2_reactions=None,
                           fname_ternary_o2_reactions=None,
                           fname_binary_oxide_formations=None,
                           fname_ternary_oxide_formations=None,
                           metal_corrections=None,
                           metal_corrections_fit=False, weighted=False,
                           l2reg=None, counter=[0]):

        if metal_corrections is not None:
            self.metal_corrections = metal_corrections
        if metal_corrections_fit:
            self.fit_metal_corrections(U)

        if self.metal_corrections is not None:
            Jain = [self.metal_corrections[M] for M in self.TM_species]
        else:
            Jain = [0.0 for M in self.TM_species]

        self.errors = {}
        if binary_oxide_reactions:
            s = 'Binary oxide reactions'
            fname = fname_binary_oxide_reactions
            self.errors[s] = self.reactions[s].reaction_energy_rmse_mae(
                U, weighted=weighted)
            if fname is not None:
                self.reactions[s].print_reactions(U, fname)

        if ternary_oxide_reactions:
            s = 'Ternary oxide reactions'
            fname = fname_ternary_oxide_reactions
            self.errors[s] = self.reactions[s].reaction_energy_rmse_mae(
                U, weighted=weighted)
            if fname is not None:
                self.reactions[s].print_reactions(U, fname)

        if binary_o2_reactions:
            s = 'Binary O2 reactions'
            fname = fname_binary_o2_reactions
            self.errors[s] = self.reactions[s].reaction_energy_rmse_mae(
                U, weighted=weighted)
            if fname is not None:
                self.reactions[s].print_reactions(U, fname)

        if ternary_o2_reactions:
            s = 'Ternary O2 reactions'
            fname = fname_ternary_o2_reactions
            self.errors[s] = self.reactions[s].reaction_energy_rmse_mae(
                U, weighted=weighted)
            if fname is not None:
                self.reactions[s].print_reactions(U, fname)

        if binary_oxide_formations:
            s = 'Binary oxide formation energies'
            fname = fname_binary_oxide_formations
            self.errors[s] = self.reactions[s].reaction_energy_rmse_mae(
                U, Jain=Jain, weighted=weighted)
            if fname is not None:
                self.reactions[s].print_formation_energies(
                    U, fname, Jain=Jain)

        if ternary_oxide_formations:
            s = 'Ternary oxide formation energies'
            fname = fname_ternary_oxide_formations
            self.errors[s] = self.reactions[s].reaction_energy_rmse_mae(
                U, Jain=Jain, weighted=weighted)
            if fname is not None:
                self.reactions[s].print_formation_energies(
                    U, fname, Jain=Jain)

        if verbose:
            self.print_errors()

        # loss function to achieve similar errors for all targets
        # rmse = [e[0] for e in self.errors.values()]
        # loss = np.sum(rmse) + np.std(rmse)

        rmse = 0.0
        N_tot = 0
        for kind in self.reactions:
            N = len(self.reactions[kind].reactions)
            rmse += N*self.errors[kind][0]**2
            N_tot += N
        rmse = np.sqrt(rmse/N_tot)
        loss = rmse

        # R, dE_exp, reactions, num_atoms = self.reaction_matrix()
        # E_comp = self.compound_energies(U, Jain=Jain)
        # err = (R.dot(E_comp) - dE_exp)/num_atoms
        # rmse2 = np.sqrt(np.mean(err**2))
        # print("CHECK: ", rmse, rmse2)

        if l2reg is not None:
            norm = np.linalg.norm(U)
            # if metal_corrections is not None:
            #     norm += np.linalg.norm(list(metal_corrections.values()))
            loss += l2reg*norm

        if print_iterations:
            counter[0] += 1
            if self.metal_corrections is not None:
                print("{:4d} : ".format(counter[0])
                      + (len(U)*"{:4.2f} ").format(*U)
                      + (len(U)*"{:4.2f} ").format(
                          *list(self.metal_corrections.values()))
                      + ": {} {} {}".format(loss, np.sum(rmse), np.std(rmse)))
            else:
                print("{:4d} : ".format(counter[0])
                      + (len(U)*"{:4.2f} ").format(*U)
                      + ": {}".format(loss))
        return loss

    def optimize_U(self, U, U_val_max, metal_corrections=None,
                   metal_corrections_fit=False,
                   print_iterations=False, l2reg=None,
                   opt_binary_oxide_reactions=False,
                   opt_ternary_oxide_reactions=False,
                   opt_binary_o2_reactions=False,
                   opt_ternary_o2_reactions=False,
                   opt_binary_oxide_formations=False,
                   opt_ternary_oxide_formations=False):
        """
        Least squares optimization of the U values.

        Returns:
          U_opt (list): optimized U values

        """

        if not any([opt_binary_oxide_reactions,
                    opt_ternary_oxide_reactions,
                    opt_binary_o2_reactions,
                    opt_ternary_o2_reactions,
                    opt_binary_oxide_formations,
                    opt_ternary_oxide_formations]):
            return np.array(U)
        else:
            def error_func(U):
                return self.eval_energy_errors(
                    U, metal_corrections=metal_corrections,
                    metal_corrections_fit=metal_corrections_fit,
                    weighted=False, print_iterations=print_iterations,
                    l2reg=l2reg,
                    binary_oxide_reactions=opt_binary_oxide_reactions,
                    ternary_oxide_reactions=opt_ternary_oxide_reactions,
                    binary_o2_reactions=opt_binary_o2_reactions,
                    ternary_o2_reactions=opt_ternary_o2_reactions,
                    binary_oxide_formations=opt_binary_oxide_formations,
                    ternary_oxide_formations=opt_ternary_oxide_formations)

            U_min = np.zeros(len(U))
            U_max = np.ones(len(U))*U_val_max
            if self.loocv:
                for i in range(len(self.loocv_comp)):
                    comp = self.loocv_comp[i].composition
                    comp = comp.formula.replace(" ", "")
                    print("\n>>>> LOO-CV Composition: {} <<<<".format(comp))
                    self.reactions = self.reactions_one_out[i]
                    results = least_squares(error_func, U,
                                            bounds=(U_min, U_max),
                                            ftol=1.0e-4, xtol=1.0e-3)
                    U_opt = results.x
                    self.print_U_values(U_opt)
                    self.print_Jain()
                    self.reactions = self.reactions_one_in[i]
                    dirname = "loocv-{}".format(comp)
                    os.mkdir(dirname)
                    self.eval_energy_errors(
                        U_opt, verbose=True,
                        fname_binary_oxide_reactions=os.path.join(
                            dirname, "oxide-reactions-2.dat"),
                        fname_ternary_oxide_reactions=os.path.join(
                            dirname, "oxide-reactions-3.dat"),
                        fname_binary_o2_reactions=os.path.join(
                            dirname, "O2-reactions-2.dat"),
                        fname_ternary_o2_reactions=os.path.join(
                            dirname, "O2-reactions-3.dat"),
                        fname_binary_oxide_formations=os.path.join(
                            dirname, "formation-energies-2.dat"),
                        fname_ternary_oxide_formations=os.path.join(
                            dirname, "formation-energies-3.dat"))
                self.reactions = self.all_reactions

            results = least_squares(error_func, U, bounds=(U_min, U_max),
                                    ftol=1.0e-4, xtol=1.0e-3)

            if not results.success:
                raise ValueError("U fit not converged.")

            U_opt = results.x
            return U_opt

    def __str__(self):
        return
