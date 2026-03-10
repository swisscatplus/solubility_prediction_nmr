import pandas as pd
from fastsolv import fastsolv
from controller.predictor_runner import process_sample_file, solubility_file_matrix, compare_predictions
from data_converter.hplc_data_handler import NMR_SOLVENTS, prepare_fastsolv_input, cleaning_array_by_cas
from analysis.analysis import (count_over_under_estimation, count_incoherence_molecule, bar_plot_over_under_estimation, butterfly_plot_over_under_estimation, 
plot_ordered_solubility_array, analyze_plot_chemical_bias, plot_rt_ordered_solubility, plot_master_overlaid_multi_trends, plot_solvent_correlation_heatmap,
plot_mw_error_distribution, plot_hydrogen_bonding_bias)


def main():


    

    """df_with_smiles = process_sample_file("/Users/arthurbenard/Project 1B/data/Fichier final (RT+sol).xlsx")
    df_ready = prepare_fastsolv_input(df_with_smiles)
    final_results = solubility_file_matrix(df_ready, NMR_SOLVENTS)"""

    """compare_predictions('/Users/arthurbenard/Project 1B/data/Master_Solubility_Matrix.xlsx','/Users/arthurbenard/Project 1B/data/Fichier final (RT+sol).xlsx', NMR_SOLVENTS)"""
    """count_over_under_estimation('/Users/arthurbenard/Project 1B/data/Compared_Results.xlsx',NMR_SOLVENTS)
    count_incoherence_molecule('/Users/arthurbenard/Project 1B/data/Compared_Results.xlsx',NMR_SOLVENTS)"""



    """count = count_over_under_estimation('/Users/arthurbenard/Project 1B/data/Compared_Results.xlsx',NMR_SOLVENTS)
    bar_plot_over_under_estimation(count)
    butterfly_plot_over_under_estimation(count)"""
    
    """plot_ordered_solubility_array('/Users/arthurbenard/Project 1B/data/Compared_Results.xlsx', 'CHCl3')"""
    
    """plot_rt_ordered_solubility('/Users/arthurbenard/Project 1B/data/Compared_Results.xlsx', 'CHCl3')"""
    """plot_master_overlaid_multi_trends('/Users/arthurbenard/Project 1B/data/Compared_Results.xlsx',NMR_SOLVENTS,'CHCl3')"""
    """plot_solvent_correlation_heatmap('/Users/arthurbenard/Project 1B/data/Compared_Results.xlsx',NMR_SOLVENTS)"""
    
    
   
    """for solvent in NMR_SOLVENTS.keys():
        print(f"\n=========================================")
        print(f"   Analyzing Molecular Weight for {solvent}")
        print(f"=========================================")
        
      
        plot_mw_error_distribution('/Users/arthurbenard/Project 1B/data/Compared_Results.xlsx',solvent)"""
    
    """for solvent in NMR_SOLVENTS.keys():
        print(f"\n=========================================")
        analyze_plot_chemical_bias('/Users/arthurbenard/Project 1B/data/Compared_Results.xlsx',solvent)"""
    """plot_hydrogen_bonding_bias('/Users/arthurbenard/Project 1B/data/Compared_Results.xlsx',solvent)"""

    


if __name__ == "__main__":
    # The guard is still necessary for the multiprocessing part
    main()