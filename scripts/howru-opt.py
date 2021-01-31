#!/usr/bin/env python

"""
Optimize Hubbard U values for DFT+U

"""

import argparse
import json
import numpy as np

from howru.optimize import UOptimizer

__author__ = "Nongnuch Artrith, Alexander Urban"
__email__ = "nartrith@atomistic.net, aurban@atomistic.net"
__date__ = "2020-07-23"
__version__ = "0.1"


def optimize_U_values(U, TM_species, metal_corrections, elements_csv,
                      atoms_csv, dimers_csv, binary_oxides_csv,
                      ternary_oxides_csv, opt_oxides_2, opt_oxides_3,
                      opt_o2_2, opt_o2_3, opt_formation_2,
                      opt_formation_3, opt_dimer, verbose,
                      U_val_max=10.0, Jain_correction=False,
                      iterative_Jain_fit=False, l2reg=None,
                      loocv=False):

    if iterative_Jain_fit:
        Jain_correction = True

    optimizer = UOptimizer(TM_species, elements_csv, atoms_csv,
                           dimers_csv, binary_oxides_csv,
                           ternary_oxides_csv, loocv=loocv)

    print("Initial Jain correction:")
    optimizer.print_Jain(metal_corrections)
    print("\nInitial U values:")
    optimizer.print_U_values(U)

    U_opt = optimizer.optimize_U(U, U_val_max, print_iterations=verbose,
                                 metal_corrections=metal_corrections,
                                 metal_corrections_fit=iterative_Jain_fit,
                                 l2reg=l2reg,
                                 opt_binary_oxide_reactions=opt_oxides_2,
                                 opt_ternary_oxide_reactions=opt_oxides_3,
                                 opt_binary_o2_reactions=opt_o2_2,
                                 opt_ternary_o2_reactions=opt_o2_3,
                                 opt_binary_oxide_formations=opt_formation_2,
                                 opt_ternary_oxide_formations=opt_formation_3,
                                 opt_dimer_binding_energies=opt_dimer)

    if any(U_opt != U):
        print("\nOptimized U values:")
        optimizer.print_U_values(U_opt)

    print()
    print("Errors with initial Jain correction:")
    optimizer.eval_energy_errors(
        U, verbose=True,
        fname_binary_oxide_reactions="oxide-reactions-2.dat",
        fname_ternary_oxide_reactions="oxide-reactions-3.dat",
        fname_binary_o2_reactions="O2-reactions-2.dat",
        fname_ternary_o2_reactions="O2-reactions-3.dat",
        fname_binary_oxide_formations="formation-energies-2.dat",
        fname_ternary_oxide_formations="formation-energies-3.dat",
        fname_dimer_binding_energies="dimer-energies.dat",
        metal_corrections=metal_corrections)

    if Jain_correction:
        optimizer.fit_metal_corrections(U_opt)
        print()
        print("Errors with optimized Jain correction:")
        print("  " + (len(TM_species)*"{:5s} ").format(*list(TM_species)))
        print(" [" + ", ".join(["{:5.2f}".format(c).strip()
                                for c in optimizer.metal_corrections.values()
                                ]) + "]")
        optimizer.eval_energy_errors(
            U_opt, verbose=True,
            binary_oxide_reactions=False,
            ternary_oxide_reactions=False,
            binary_o2_reactions=False,
            ternary_o2_reactions=False,
            fname_binary_oxide_formations="formation-energies-2-jain.dat",
            fname_ternary_oxide_formations="formation-energies-3-jain.dat",
            fname_dimer_binding_energies="dimer-energies-jain.dat",
            metal_corrections=optimizer.metal_corrections)


def main():

    parser = argparse.ArgumentParser(
        description=__doc__+"\n{} {}".format(__date__, __author__),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "input_file",
        help="Input file in JSON format")

    args = parser.parse_args()

    with open(args.input_file) as fp:
        options = json.load(fp)

    TM_species = np.array(options["species"])
    metal_corrections = dict(zip(TM_species, options["jain_corrections"]))

    def tryoption(key):
        if key in options:
            return options[key]
        else:
            return None

    optimize_U_values(
        options["u_values"],
        TM_species,
        metal_corrections,
        options["elements"],
        options["atoms"],
        options["dimers"],
        options["binary_oxides"],
        options["ternary_oxides"],
        options["optimize_binary_oxides"],
        options["optimize_ternary_oxides"],
        options["optimize_binary_o2_reactions"],
        options["optimize_ternary_o2_reactions"],
        options["optimize_binary_formation_reactions"],
        options["optimize_ternary_formation_reactions"],
        options["optimize_dimer_binding"],
        options["verbose"],
        Jain_correction=options["optimize_jain_correction"],
        iterative_Jain_fit=options["iterative_jain_correction_fit"],
        l2reg=tryoption("l2_regularization"),
        loocv=tryoption("leave_one_out_cv"))


if (__name__ == "__main__"):
    main()
