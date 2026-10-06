"""
fastsolv analysis pipeline (logS predictions vs experimental solubility).

Run from the project root:   python src/main.py
Uncomment the steps you want to run. Only the chemical-bias step is active,
exactly as in the original version.
"""
import pandas as pd
from fastsolv import fastsolv
from config import HPLC_RT_SOLUBILITY, FASTSOLV_PREDICTIONS, FASTSOLV_VS_EXPERIMENT
from controller.predictor_runner import process_sample_file, solubility_file_matrix, compare_predictions
from data_converter.hplc_data_handler import NMR_SOLVENTS, prepare_fastsolv_input, cleaning_array_by_cas
from analysis.analysis import (count_over_under_estimation, count_incoherence_molecule, bar_plot_over_under_estimation, butterfly_plot_over_under_estimation,
plot_ordered_solubility_array, analyze_plot_chemical_bias, plot_rt_ordered_solubility, plot_master_overlaid_multi_trends, plot_solvent_correlation_heatmap,
plot_mw_error_distribution, plot_hydrogen_bonding_bias)


def main():

    # --- 1. SMILES lookup + fastsolv predictions -> data/interim/fastsolv_predictions.xlsx
    # df_with_smiles = process_sample_file(HPLC_RT_SOLUBILITY)
    # df_ready = prepare_fastsolv_input(df_with_smiles)
    # final_results = solubility_file_matrix(df_ready, NMR_SOLVENTS)

    # --- 2. Compare with experiment -> data/interim/fastsolv_vs_experiment.xlsx
    # compare_predictions(FASTSOLV_PREDICTIONS, HPLC_RT_SOLUBILITY, NMR_SOLVENTS)

    # --- 3. Counts and plots (figures -> results/figures/)
    # count_over_under_estimation(FASTSOLV_VS_EXPERIMENT, NMR_SOLVENTS)
    # count_incoherence_molecule(FASTSOLV_VS_EXPERIMENT, NMR_SOLVENTS)

    # count = count_over_under_estimation(FASTSOLV_VS_EXPERIMENT, NMR_SOLVENTS)
    # bar_plot_over_under_estimation(count)
    # butterfly_plot_over_under_estimation(count)

    # plot_ordered_solubility_array(FASTSOLV_VS_EXPERIMENT, 'CHCl3')
    # plot_rt_ordered_solubility(FASTSOLV_VS_EXPERIMENT, 'CHCl3')
    # plot_master_overlaid_multi_trends(FASTSOLV_VS_EXPERIMENT, NMR_SOLVENTS, 'CHCl3')
    # plot_solvent_correlation_heatmap(FASTSOLV_VS_EXPERIMENT, NMR_SOLVENTS)

    # for solvent in NMR_SOLVENTS.keys():
    #     print(f"\n=========================================")
    #     print(f"   Analyzing Molecular Weight for {solvent}")
    #     print(f"=========================================")
    #     plot_mw_error_distribution(FASTSOLV_VS_EXPERIMENT, solvent)

    for solvent in NMR_SOLVENTS.keys():
        print(f"\n=========================================")
        analyze_plot_chemical_bias(FASTSOLV_VS_EXPERIMENT, solvent)
        # plot_hydrogen_bonding_bias(FASTSOLV_VS_EXPERIMENT, solvent)


if __name__ == "__main__":
    # The guard is still necessary for the multiprocessing part
    main()
