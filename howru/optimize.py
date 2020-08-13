"""
Class implementing the U value optimization.

"""

import os
import numpy as np
from scipy.optimize import least_squares
# from scipy.optimize import minimize

from .compound import read_compounds_from_csv, read_elements_from_csv
from .reactions import Reactions

__date__ = "2020-07-23"
__version__ = "0.1"


class UOptimizer(object):

    def __init__(self, TM_species, elements_csv, atoms_csv, dimers_csv,
                 binary_oxides_csv, ternary_oxides_csv, loocv=False):

        self.TM_species = TM_species
        self.elements = read_elements_from_csv(elements_csv, TM_species)
        self.atoms = read_elements_from_csv(atoms_csv, TM_species)
        self.dimers = read_compounds_from_csv(dimers_csv, TM_species)
        self.binary_oxides = read_compounds_from_csv(
            binary_oxides_csv, TM_species)
        self.ternary_oxides = read_compounds_from_csv(
            ternary_oxides_csv, TM_species)

        self.U = np.zeros(len(self.TM_species))
        self.metal_corrections = None

        self.reactions = Reactions(self.elements, self.atoms,
                                   self.dimers, self.binary_oxides,
                                   self.ternary_oxides)

        self.loocv = loocv
        if loocv:
            self.reactions_one_out = []
            self.reactions_one_in = []
            self.loocv_comp = []
            self.all_reactions = self.reactions
            N = len(self.binary_oxides)
            for i in range(N):
                idx = [j for j in range(N) if j != i]
                self.loocv_comp.append(self.binary_oxides[i])
                self.reactions_one_out.append(
                    Reactions(self.elements, self.atoms,
                              self.dimers,
                              [self.binary_oxides[j] for j in idx],
                              self.ternary_oxides))
                self.reactions_one_in.append(
                    Reactions(self.elements, self.atoms,
                              self.dimers, self.binary_oxides,
                              self.ternary_oxides,
                              required_compounds=[self.loocv_comp[-1]]))
            N = len(self.ternary_oxides)
            for i in range(N):
                idx = [j for j in range(N) if j != i]
                self.reactions_one_out.append(
                    Reactions(self.elements, self.atoms,
                              self.dimers, self.binary_oxides,
                              [self.ternary_oxides[j] for j in idx]))
                self.reactions_one_in.append(
                    Reactions(self.elements, self.atoms,
                              self.dimers, self.binary_oxides,
                              self.ternary_oxides,
                              required_compounds=[self.ternary_oxides[i]]))

        self.errors = {}

    def fit_metal_corrections(self, U):
        """
        Fit the metal correction energy term following Jain et al. for all
        TM species.

        Arguments:
          U (list): list of U values

        """
        metal_corrections = {}
        for M in self.TM_species:
            M_oxides = [c for c in self.binary_oxides if M in c.composition]
            if len(M_oxides) == 0:
                metal_corrections[M] = 0.0
            else:
                _, (comp, Ef_expt, Ef_calc
                    ) = self.reactions.eval_formation_energies(M_oxides, U)
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

    def calc_errors(self, reactions, weighted=False):
        """
        Calculate reaction energy errors.

        Arguments:
          reactions (dict): Reaction dictionary returned by
            reactions.Reactions.get_reactions()
          weighted (bool): If True, multiply errors by number of reactions

        Returns:
          np.array([rmse, mae])

        """
        if len(reactions) == 0:
            return np.array([0.0, 0.0])
        rmse = 0.0
        mae = 0.0
        for r in reactions:
            dEr_comp, dEr_expt = reactions[r]
            error = (dEr_expt - dEr_comp)/self.reactions.num_atoms(r)
            mae += abs(error)
            rmse += error*error
        mae /= len(reactions)
        rmse = np.sqrt(rmse/len(reactions))
        w = len(reactions) if weighted else 1
        return np.array([w*rmse, w*mae])

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
        print("      " + (len(self.TM_species)*"{:5s} "
                          ).format(*list(self.TM_species)))
        print(" c = [" + ", ".join(
            ["{:5.2f}".format(c).strip()
             for c in m.values()]) + "]")

    def eval_energy_errors(self, U, verbose=False,
                           print_iterations=False,
                           binary_oxide_reactions=True,
                           ternary_oxide_reactions=True,
                           binary_o2_reactions=True,
                           ternary_o2_reactions=True,
                           binary_oxide_formations=True,
                           ternary_oxide_formations=True,
                           dimer_binding_energies=True,
                           fname_binary_oxide_reactions=None,
                           fname_ternary_oxide_reactions=None,
                           fname_binary_o2_reactions=None,
                           fname_ternary_o2_reactions=None,
                           fname_binary_oxide_formations=None,
                           fname_ternary_oxide_formations=None,
                           fname_dimer_binding_energies=None,
                           metal_corrections=None,
                           metal_corrections_fit=False, weighted=False,
                           l2reg=None, counter=[0]):

        if metal_corrections is not None:
            self.metal_corrections = metal_corrections
        if metal_corrections_fit:
            self.fit_metal_corrections(U)

        self.errors = {}
        if binary_oxide_reactions:
            self.reactions.eval_binary_oxide_reactions(
                U, fname=fname_binary_oxide_reactions)
            self.errors['Binary oxide reactions'] = (
                self.calc_errors(self.reactions.binary_oxide_reactions,
                                 weighted=weighted))
        if ternary_oxide_reactions:
            self.reactions.eval_ternary_oxide_reactions(
                U, fname=fname_ternary_oxide_reactions)
            self.errors['Ternary oxide reactions'] = (
                self.calc_errors(self.reactions.ternary_oxide_reactions,
                                 weighted=weighted))
        if binary_o2_reactions:
            self.reactions.eval_binary_o2_reactions(
                U, fname=fname_binary_o2_reactions)
            self.errors['Binary O2 reactions'] = (
                self.calc_errors(self.reactions.binary_o2_reactions,
                                 weighted=weighted))
        if ternary_o2_reactions:
            self.reactions.eval_ternary_o2_reactions(
                U, fname=fname_ternary_o2_reactions)
            self.errors['Ternary O2 reactions'] = (
                self.calc_errors(self.reactions.ternary_o2_reactions,
                                 weighted=weighted))
        if binary_oxide_formations:
            self.reactions.eval_binary_oxide_formations(
                U, metal_corrections=self.metal_corrections,
                fname=fname_binary_oxide_formations)
            self.errors['Binary oxide formation energies'] = (
                self.calc_errors(self.reactions.binary_oxide_formations,
                                 weighted=weighted))
        if ternary_oxide_formations:
            self.reactions.eval_ternary_oxide_formations(
                U, metal_corrections=self.metal_corrections,
                fname=fname_ternary_oxide_formations)
            self.errors['Ternary oxide formation energies'] = (
                self.calc_errors(self.reactions.ternary_oxide_formations,
                                 weighted=weighted))
        if dimer_binding_energies:
            self.reactions.eval_dimer_binding_energies(
                U, metal_corrections=self.metal_corrections,
                fname=fname_dimer_binding_energies)
            self.errors['Dimer binding energies'] = (
                self.calc_errors(self.reactions.dimer_binding_energies,
                                 weighted=weighted))
        if verbose:
            self.print_errors()

        error = 0.0
        for e in self.errors:
            # sum of RMSEs
            error += self.errors[e][0]

        if l2reg is not None:
            norm = np.linalg.norm(U)
            if metal_corrections is not None:
                norm += np.linalg.norm(list(metal_corrections.values()))
            error += l2reg*norm

        if print_iterations:
            counter[0] += 1
            if self.metal_corrections is not None:
                print("{:4d} : ".format(counter[0])
                      + (len(U)*"{:4.2f} ").format(*U)
                      + (len(U)*"{:4.2f} ").format(
                          *list(self.metal_corrections.values()))
                      + ": {}".format(error))
            else:
                print("{:4d} : ".format(counter[0])
                      + (len(U)*"{:4.2f} ").format(*U)
                      + ": {}".format(error))
        return error

    def optimize_U(self, U, U_val_max, metal_corrections=None,
                   metal_corrections_fit=False,
                   print_iterations=False, l2reg=None,
                   opt_binary_oxide_reactions=False,
                   opt_ternary_oxide_reactions=False,
                   opt_binary_o2_reactions=False,
                   opt_ternary_o2_reactions=False,
                   opt_binary_oxide_formations=False,
                   opt_ternary_oxide_formations=False,
                   opt_dimer_binding_energies=False):
        """
        Least squares optimization of the U values.

        """

        if not any([opt_binary_oxide_reactions,
                    opt_ternary_oxide_reactions,
                    opt_binary_o2_reactions,
                    opt_ternary_o2_reactions,
                    opt_binary_oxide_formations,
                    opt_ternary_oxide_formations,
                    opt_dimer_binding_energies]):
            return
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
                    ternary_oxide_formations=opt_ternary_oxide_formations,
                    dimer_binding_energies=opt_dimer_binding_energies)

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
                                            ftol=1.0e-3, xtol=1.0e-2)
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
                                    ftol=1.0e-3, xtol=1.0e-2)

            # Alternatively, use SciPy's minimize function:
            # U_bound = [(0, U_val_max) for i in range(len(U))]
            # results = minimize(error_func, U, bounds=U_bound, tol=1.0e-3)

            if not results.success:
                raise ValueError("U fit not converged.")

            U_opt = results.x
            return U_opt

    def __str__(self):
        return
