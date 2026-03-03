import pandas as pd
from fastsolv import fastsolv
from controller.predictor_runner import process_sample_file
from controller.predictor_runner import solubility_file_matrix
from controller.predictor_runner import compare_predictions
from data_converter.hplc_data_handler import NMR_SOLVENTS
from data_converter.hplc_data_handler import prepare_fastsolv_input
from analysis.analysis import count_over_under_estimation


def main():


    

    """df_with_smiles = process_sample_file("/Users/arthurbenard/Project 1B/data/Fichier final (RT+sol).xlsx")
    df_ready = prepare_fastsolv_input(df_with_smiles)
    final_results = solubility_file_matrix(df_ready, NMR_SOLVENTS)"""

    """ compare_predictions('/Users/arthurbenard/Project 1B/data/Master_Solubility_Matrix.xlsx', '/Users/arthurbenard/Project 1B/data/Fichier final (RT+sol).xlsx', NMR_SOLVENTS)"""
    count_over_under_estimation('/Users/arthurbenard/Project 1B/data/Compared_Results.xlsx',NMR_SOLVENTS)

if __name__ == "__main__":
    # The guard is still necessary for the multiprocessing part
    main()