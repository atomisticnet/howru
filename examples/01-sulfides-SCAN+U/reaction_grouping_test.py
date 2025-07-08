#!/usr/bin/env python3
"""
Script to determine optimal U-Values and Jain-Values with an option for LOOCV for sulfides
"""

import os
import sys
import math
import functools
import importlib

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
sys.path.insert(0, script_dir) 

from util import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from optimize import UOptimizer


try:
    from pymatgen.ext.matproj import MPRester
except ImportError:
    print("MPRester could not be imported from pymatgen")



def read_compounds_from_csv(filepath):
    """Read compound records from a CSV file."""
    return pd.read_csv(filepath).to_dict(orient='records')


def read_elements_from_csv(filepath, tm_species):
    """Read element records from a CSV file."""
    return pd.read_csv(filepath).to_dict(orient='records')


def has_e_expt(compounds):
    """Filter compounds with experimental formation energies."""
    return [
        c for c in compounds
        if isinstance(c, dict) and 'expt_formation_energy_RT' in c
        and not np.isnan(c['expt_formation_energy_RT'])
    ]

def calculate_similarity(R):
    """Calculate cosine similarity weights from matrix R."""
    similarities = cosine_similarity(R)
    weights = np.sum(similarities, axis=1)
    return weights


def reaction_grouping():
    """
    Determine optimal Hubbard U and Jain correction parameters using DFT 
    energies for elemental, binary, and ternary sulfides.

    This function reads hard-coded CSV files containing DFT energies, then 
    employs least-squares optimization with L2 regularization
    to compute the optimal Hubbard U and Jain parameters. The resulting errors 
    are printed and detailed results are saved to output files.

    Returns:
    results_df: A DataFrame containing l2 regularization 
    values, RMSE, U values, and Jain corrections for each optimization run.
    """
    elements_path = "BCD4 - Calculated Energies.csv"
    binaries_path = "BCD4 - Calculated Energies - Binary Sulfides.csv"
    ternaries_path = "BCD4 - Calculated Energies - Ternary Sulfides.csv"
    tm_species = ['Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Sc', 'Zn']

    elements = read_elements_from_csv(elements_path, tm_species)

    opt = UOptimizer(tm_species, elements_path, binaries_path, ternaries_path)
    R, dE_exp, reactions, num_atoms = opt.reaction_matrix()
    weights = calculate_similarity(R)

    U = [0.0] * len(tm_species)
    E_comp = opt.compound_energies(U, Jain='auto')
    err = R.dot(E_comp) - dE_exp

    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))

    print(f"mae (no +U) = {mae:.6f} eV")
    print(f"rmse (no +U) = {rmse:.6f} eV")

    with open('optimization_results.txt', 'w') as file:
        file.write(
            f"Mean Absolute Error (MAE) without U values: {mae:.6f} eV\n"
        )
        file.write(
            f"Root Mean Square Error (RMSE) without U values: {rmse:.6f} eV\n"
        )

    min_rmse = np.inf
    min_l2 = None
    min_U = None
    results_df = pd.DataFrame(columns=['l2', 'rmse', 'U', 'J'])

    def objective(U, l2):
        R, dE_exp, reactions, num_atoms = opt.reaction_matrix()
        E_comp = opt.compound_energies(U, Jain='auto')
        err = R.dot(E_comp) - dE_exp
        return np.sqrt(np.mean(err ** 2)) + l2 * np.linalg.norm(U)

    # Initial guess for U-values
    U = [0.28, 0.02, 0.04, 0.51, 0.12, 0.15, 0.15, 0.00, 0.00]

    df = pd.DataFrame(columns=['l2', 'rmse', 'U', 'J'])
    for l2 in [0.000, 0.005, 0.01, 0.015, 0.02, 0.025, 0.030,
               0.035, 0.040, 0.045, 0.050, 0.1, 0.2]:
        f = functools.partial(objective, l2=l2)
        results = least_squares(
            f, U, bounds=(0.0, 5.0), ftol=1.0e-4, xtol=1.0e-3
        )
        U = results.x
        R, dE_exp, reactions, num_atoms = opt.reaction_matrix()

        E_comp = opt.compound_energies(U, Jain='auto')
        if 'Zn' in opt.TM_species:
            Jain_corrections = [0.0 for _ in U]
            Jain_corrections[opt.TM_species.index('Zn')] = 0.12
        err = R.dot(E_comp) - dE_exp
        rmse = np.sqrt(np.mean(err ** 2))
        J = list(opt.metal_corrections.values())
        df = pd.concat(
            [
                df,
                pd.DataFrame([{'l2': l2, 'rmse': rmse,
                               'U': U.tolist(), 'J': J}])
            ],
            ignore_index=True
        )
        print("done with l2 = {}".format(l2))

        if rmse < min_rmse:
            min_rmse = rmse
            min_l2 = l2
            min_U = U

        temp_df = pd.DataFrame({
            'l2': [l2],
            'rmse': [rmse],
            'U': [U],
            'J': [J]
        })
        results_df = pd.concat([results_df, temp_df], ignore_index=True)
        print(f"Completed l2 = {l2} with RMSE = {rmse}")

    l2 = min_l2
    U = min_U
    f = functools.partial(objective, l2=l2)
    results = least_squares(
        f, U, bounds=(0.0, 5.0), ftol=1.0e-4, xtol=1.0e-3
    )
    U_opt, J_opt = results.x[:len(tm_species)], results.x[len(tm_species):]
    R, dE_exp, reactions, num_atoms = opt.reaction_matrix()
    E_comp = opt.compound_energies(U, Jain='auto')
    err = R.dot(E_comp) - dE_exp
    rmse = np.sqrt(np.mean(err ** 2))
    print('rmse (with +U) = {:.6f} eV'.format(rmse))
    print("mae (with +U) = {:.6f} eV".format(np.mean(np.abs(err))))
    if 'Zn' in opt.metal_corrections:
        opt.metal_corrections['Zn'] = 0.12
        print("Setting Zn Jain correction to 0.12")
    print(
        '        ',
        (len(U) * ' {:5s}').format(*opt.metal_corrections.keys())
    )
    print(
        'U_opt = ',
        (len(U) * '{:5.2f} ').format(*U_opt)
    )
    print(
        'J_opt = ',
        (len(U) * '{:5.2f} ').format(*opt.metal_corrections.values())
    )
    results_df.to_csv('l2-regularization.csv', index=False)
    with open('optimization_results.txt', 'a') as file:
        file.write(
            f"Mean Absolute Error (MAE) with U values: "
            f"{np.mean(np.abs(err)):.6f} eV\n"
        )
        file.write(
            f"Root Mean Square Error (RMSE) with U values: "
            f"{rmse:.6f} eV\n"
        )
        file.write(
            'U_opt = ' + (len(U) * '{:5.2f} ').format(*U_opt) + '\n'
        )
        file.write(
            'J_opt = ' +
            (len(U) * '{:5.2f} ').format(*opt.metal_corrections.values()) +
            '\n'
        )

    return results_df



def reaction_grouping_loocv():
    """
    Hubbard-U uncertainty determination via LOOCV for sulfides.
    This function uses hard-coded CSV files containing DFT energies as inputs.
    It returns a CSV file and a text file with the LOOCV results.
    
    Returns:
    opt: UOptimizer instance with LOOCV results and saved output files.
    """
    elements_path = "BCD4 - Calculated Energies.csv"
    binaries_path = "BCD4 - Calculated Energies - Binary Sulfides.csv"
    ternaries_path = "BCD4 - Calculated Energies - Ternary Sulfides.csv"
    TM_species = ['Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Sc', 'Zn']

    opt = UOptimizer(
        TM_species, elements_path, binaries_path, ternaries_path, loocv=True
    )

    loocv_results = []

    U_initial = [2.16, 0.95, 2.52, 2.29, 0.55, 2.13, 0.47, 0.5, 0]
    l2_reg = 0.01
    Jain = 'auto'

    original_stdout = sys.stdout
    sys.stdout = open("loocv_output.txt", "w")

    for i in range(len(opt.loocv_comp)):
        comp = opt.loocv_comp[i].composition.formula.replace(" ", "")
        print("\n>>>> LOO-CV Composition: {} <<<<".format(comp))

        opt.reactions = opt.reactions_one_out[i]
        R, dE_exp, reactions, num_atoms = opt.reaction_matrix()

        n_samples = R.shape[0]
        n_components = min(25, n_samples)
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(R)

        n_clusters = min(40, n_samples) if n_samples > 0 else 1
        kmeans = KMeans(
            n_clusters=n_clusters, n_init=10, random_state=42
        ).fit(components)
        labels = kmeans.labels_
        label_counts = {
            label: list(labels).count(label) for label in set(labels)
        }
        weights = np.array([
            1.0 / label_counts[label] for label in labels
        ])

        def objective_loocv(U):
            E_comp = opt.compound_energies(U, Jain=Jain)
            err = (R.dot(E_comp) - dE_exp) * weights
            a = len(weights) / np.sum(weights)
            return np.sqrt(np.mean(err**2)) * a + l2_reg * np.linalg.norm(U)

        results = least_squares(
            lambda U: objective_loocv(U), U_initial, bounds=(0.0, 5.0),
            ftol=1.0e-4, xtol=1.0e-3
        )
        U_opt = results.x
        print("Optimized U for training set:", U_opt)

        opt.reactions = opt.reactions_one_in[i]
        R_val, dE_exp_val, reactions_val, num_atoms_val = opt.reaction_matrix()
        E_comp_val = opt.compound_energies(U_opt, Jain=Jain)
        err_val = R_val.dot(E_comp_val) - dE_exp_val
        rmse_val = np.sqrt(np.mean(err_val**2))
        mae_val = np.mean(np.abs(err_val))
        print(
            f"Left-out compound {comp} evaluation: RMSE = {rmse_val:.6f} eV, "
            f"MAE = {mae_val:.6f} eV"
        )

        dirname = f"loocv-{comp}"
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        opt.eval_energy_errors(
            U_opt, verbose=True,
            fname_binary_oxide_reactions=os.path.join(
                dirname, "oxide-reactions-2.dat"
            ),
            fname_ternary_oxide_reactions=os.path.join(
                dirname, "oxide-reactions-3.dat"
            ),
            fname_binary_o2_reactions=os.path.join(
                dirname, "O2-reactions-2.dat"
            ),
            fname_ternary_o2_reactions=os.path.join(
                dirname, "O2-reactions-3.dat"
            ),
            fname_binary_oxide_formations=os.path.join(
                dirname, "formation-energies-2.dat"
            ),
            fname_ternary_oxide_formations=os.path.join(
                dirname, "formation-energies-3.dat"
            )
        )

        opt.reactions = opt.all_reactions

        loocv_results.append({
            'compound': comp,
            'U_opt': U_opt.tolist(),
            'rmse': rmse_val,
            'mae': mae_val
        })

        U_initial = U_opt

    sys.stdout.close()
    sys.stdout = original_stdout

    df_loocv = pd.DataFrame(loocv_results)
    df_loocv.to_csv("loocv_results.csv", index=False)
    print("LOOCV evaluation completed. Results saved to loocv_results.csv.")

    return opt


def plot_correlation():
    """
    Reads the optimal U CSV file and computes error metrics for formation
    energies per atom and per sulfur.

    Returns:
    Text output containing RMSE and MAE for energy per atom and
    energy per sulfur.
    """
    optimal_U = "BCD4 - Calculated Energies - Optimal U Value.csv"
    df = pd.read_csv(optimal_U)

    err_atom = df['Experimental Energy per Atom'] - df['Computational Energy per Atom']
    rmse_atom = math.sqrt((err_atom ** 2).mean())
    mae_atom = np.mean(np.abs(err_atom))

    err_sulfur = df['Experimental Energy per Sulfur'] - df['Computational Energy per Sulfur ']
    rmse_sulfur = math.sqrt((err_sulfur ** 2).mean())
    mae_sulfur = np.mean(np.abs(err_sulfur))
    print(rmse_atom)
    print(mae_atom)
    print(rmse_sulfur)
    print(mae_sulfur)


    return {
        "rmse_atom": rmse_atom,
        "mae_atom": mae_atom,
        "rmse_sulfur": rmse_sulfur,
        "mae_sulfur": mae_sulfur,
    }



    
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "loocv":
        print("Running LOOCV procedure for sulfides...")
        opt = reaction_grouping_loocv()
    else:
        print("Running standard optimization...")
        df = reaction_grouping()
        plot_correlation()
