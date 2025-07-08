#!/usr/bin/env python3
"""
Script to determine optimal U-Values without Jain Corrections
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import functools
import math
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import cosine_similarity
from howru.optimize import UOptimizer

try:
    from pymatgen.ext.matproj import MPRester 
except ImportError:
    print("MPRester could not be imported from pymatgen")

def read_compounds_from_csv(filepath):
    return pd.read_csv(filepath).to_dict(orient='records')

def read_elements_from_csv(filepath, TM_species):
    return pd.read_csv(filepath).to_dict(orient='records')

def has_E_expt(compounds):
    return [c for c in compounds if isinstance(c, dict) and 'expt_formation_energy_RT' in c and not np.isnan(c['expt_formation_energy_RT'])]

def calculate_similarity(R):
    similarities = cosine_similarity(R)
    weights = np.sum(similarities, axis=1)
    return weights

def reaction_grouping():
    elements_path = "BCD4 - Calculated Energies - SCAN+rvv10+U-elements (1).csv"
    binaries_path = "BCD4 - Calculated Energies - Export Data Proper Formatting Final No FeS2.csv"
    ternaries_path = "BCD4 - Calculated Energies - Ternary Sulfides (2).csv"
    TM_species = ['Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Sc', 'Zn']

    elements = read_elements_from_csv(elements_path, TM_species)
    
    opt = UOptimizer(TM_species, elements_path, binaries_path, ternaries_path)
    R, dE_exp, reactions, num_atoms = opt.reaction_matrix()
    weights = calculate_similarity(R)
    
    U = [0.0] * len(TM_species)
    E_comp = opt.compound_energies(U, Jain=None)
    err = R.dot(E_comp) - dE_exp

    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))

    print(f"mae (no +U) = {mae:.6f} eV")
    print(f"rmse (no +U) = {rmse:.6f} eV")

    with open('optimization_results.txt', 'w') as file:
        file.write(f"Mean Absolute Error (MAE) without U values: {mae:.6f} eV\n")
        file.write(f"Root Mean Square Error (RMSE) without U values: {rmse:.6f} eV\n")
        

    min_rmse = np.inf
    min_l2 = None
    min_U = None
    results_df = pd.DataFrame(columns=['l2', 'rmse', 'U'])

    def objective(U, l2):
        R, dE_exp, reactions, num_atoms = opt.reaction_matrix()
        E_comp = opt.compound_energies(U, Jain=None)
        err = (R.dot(E_comp) - dE_exp)
        return np.sqrt(np.mean(err**2)) + l2 * np.linalg.norm(U)

    U = [2.16, 0.95, 2.52, 2.29, 0.55, 2.13, 0.47, 0.5, 0.5]

    df = pd.DataFrame(columns=['l2', 'rmse', 'U'])
    for l2 in [0.000, 0.005, 0.01, 0.015, 0.02, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.1, 0.2]:
        f = functools.partial(objective, l2=l2)
        results = least_squares(f, U, bounds=(0.0, 5.0), ftol=1.0e-4, xtol=1.0e-3)
        U = results.x
        R, dE_exp, reactions, num_atoms = opt.reaction_matrix()
        E_comp = opt.compound_energies(U, Jain=None)
        err = R.dot(E_comp) - dE_exp
        rmse = np.sqrt(np.mean(err**2))
        df = pd.concat([df, pd.DataFrame([{'l2': l2, 'rmse': rmse, 'U': U.tolist()}])], ignore_index=True)

        print("done with l2 = {}".format(l2))

        if rmse < min_rmse:
            min_rmse = rmse
            min_l2 = l2
            min_U = U

        temp_df = pd.DataFrame({'l2': [l2], 'rmse': [rmse], 'U': [U]})
        results_df = pd.concat
        ([results_df, temp_df], ignore_index=True)
        print(f"Completed l2 = {l2} with RMSE = {rmse}")
        
    # Optimal U Values
    l2 = min_l2
    U = min_U
    f = functools.partial(objective, l2=l2)
    results = least_squares(f, U, bounds=(0.0, 5.0), ftol=1.0e-4, xtol=1.0e-3)
    U_opt = results.x
    R, dE_exp, reactions, num_atoms = opt.reaction_matrix()
    E_comp = opt.compound_energies(U_opt, Jain=None)
    err = R.dot(E_comp) - dE_exp
    rmse = np.sqrt(np.mean(err**2))
    print('rmse (with +U) = {:.6f} eV'.format(rmse))
    print("mae (with +U) = {:.6f} eV".format(np.mean(abs(err))))
    print('        ', (len(U)*' {:5s}').format(*opt.TM_species))
    print('U_opt = ', (len(U)*'{:5.2f} ').format(*U_opt))
    results_df.to_csv('l2-regularization.csv', index=False)
    with open('optimization_results.txt', 'a') as file:
        file.write(f"Mean Absolute Error (MAE) with U values: {np.mean(abs(err)):.6f} eV\n")
        file.write(f"Root Mean Square Error (RMSE) with U values: {rmse:.6f} eV\n")
        file.write('U_opt = ' + (len(U)*'{:5.2f} ').format(*U_opt) + '\n')          


    return results_df

def plot_l2(df):
    if isinstance(df.iloc[0]['U'], str):
        df['U'] = df['U'].apply(lambda x: [float(i) for i in x.strip("[]").split(",")])

    U_matrix = np.array(df['U'].tolist())

    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(df['l2'], df['rmse'], 'r-o', label='RMSE')
    ax[0].set_ylabel("RMSE (eV)")
    ax[0].set_title("RMSE vs L2 Regularization")
    ax[0].grid(True)
    ax[0].legend()

    metals = ['Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Sc', 'Zn']
    for i, metal in enumerate(metals):
        ax[1].plot(df['l2'], U_matrix[:, i], '-o', label=f'{metal} U')
    ax[1].set_ylabel("U Values (eV)")
    ax[1].set_xlabel("L2 Regularization")
    ax[1].set_title("U Values vs L2 Regularization")
    ax[1].legend(loc='upper right')
    ax[1].grid(True)

    plt.savefig("l2_regularization_plots_no_jain.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

def plot_correlation():
    optimal_U = "BCD4 - Calculated Energies - Optimal U Value.csv"
    df = pd.read_csv(optimal_U)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

    axes[0].scatter(df['Experimental Energy per Atom'], df['Computational Energy per Atom'], color='blue', alpha=0.7)
    axes[0].set_xlabel('Experimental Formation Energy (eV/atom)')
    axes[0].set_ylabel('Computational Formation Energy (eV/atom)')
    axes[0].set_title('Computational vs Experimental Formation Energy per Atom')
    axes[0].grid(True)

    lims = [
        np.min([axes[0].get_xlim(), axes[0].get_ylim()]),
        np.max([axes[0].get_xlim(), axes[0].get_ylim()]),
    ]
    
    axes[0].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    axes[0].set_aspect('equal')
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)

    print("Experimental and Computational Energy per Atom for each compound:")
    for idx, row in df.iterrows():
        print(f"Compound {idx+1}: Experimental Energy per Atom = {row['Experimental Energy per Atom']:.6f} eV, "
              f"Computational Energy per Atom = {row['Computational Energy per Atom']:.6f} eV")

    axes[1].scatter(df['Experimental Energy per Sulfur'], df['Computational Energy per Sulfur '], color='blue', alpha=0.7)
    axes[1].set_xlabel('Experimental Formation Energy (eV/S)')
    axes[1].set_ylabel('Computational Formation Energy (eV/S)')
    axes[1].set_title('Computational vs Experimental Formation Energy per Sulfur')
    axes[1].grid(True)
    
    lims = [
        np.min([axes[1].get_xlim(), axes[1].get_ylim()]),  
        np.max([axes[1].get_xlim(), axes[1].get_ylim()]),  
    ]
    
    axes[1].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    axes[1].set_aspect('equal')
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)

    plt.tight_layout()
    plt.savefig("correlation_plot_combined_no_jain.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    print('Correlation Plot Statistics')
    ErrDFT_atom = df['Experimental Energy per Atom'] - df['Computational Energy per Atom']
    RMSEDFT_atom = math.sqrt((ErrDFT_atom ** 2).mean())
    print('RMSE (Root Mean Square Error) for Energy per Atom: {:.6f} eV'.format(RMSEDFT_atom))

    ErrDFT_sulfur = df['Experimental Energy per Sulfur'] - df['Computational Energy per Sulfur ']
    RMSEDFT_sulfur = math.sqrt((ErrDFT_sulfur ** 2).mean())
    print('RMSE (Root Mean Square Error) for Energy per Sulfur: {:.6f} eV'.format(RMSEDFT_sulfur))

def plot_convex_hull():
    try:
        data = pd.read_csv("BCD4 - Calculated Energies - Export (5) (1).csv")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    element = 'Co'  # Change this to the element of interest

    datasets = {
        'DFT w Corrections': (f'Frac {element}', f'{element} DFT with Corrections'),
        'DFT no Corrections': (f'Frac {element}', f'{element} DFT without Corrections'),
        'Experiment': (f'Frac {element}', f'{element} Experimental Formation Energy')
    }
    
    plt.figure(figsize=(8, 6))
    for label, (x_col, y_col) in datasets.items():
        if x_col not in data.columns or y_col not in data.columns:
            print(f"Columns {x_col} or {y_col} not found in data.")
            continue
                
        points = data[[x_col, y_col]].dropna().values
        
        if points.shape[0] < 3:
            print(f"Not enough points to create a hull for {label}")
            continue
        
        hull = ConvexHull(points)
        plt.scatter(points[:, 0], points[:, 1], label=label)

        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    plt.xlabel(f'Fraction {element}')
    plt.ylabel('Formation Energy (eV/Atom)')
    plt.title(f'Convex Hull Plot for {element}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig("convex_hull_plot_combined_no_jain.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__ == "__main__":
    df = reaction_grouping()
    plot_l2(df)
    plot_correlation()
    plot_convex_hull()
