import pandas as pd
from fastsolv import fastsolv


def main():
    
    """# 1. Prepare your dataset
    # You can replace this with pd.read_csv("your_big_file.csv") later
    data = {
        "solute_name": ["Aspirin", "Caffeine"],
        "solute_smiles": ["CC(=O)Oc1ccccc1C(=O)O", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"],
        "solvent_smiles": ["CCO", "CCO"],
        "temperature": [298.15, 298.15]
    }
    df = pd.DataFrame(data)

    print("Starting FastSolv...")
    
    # 2. Run the calculation
    # We store the output in 'raw_results'
    raw_results = fastsolv(df)
    print(raw_results)

    # 3. FIX THE DATA ALIGNMENT:
    # We strip the "Tuple" index and force the numbers into a clean format
    clean_results = pd.DataFrame(
        raw_results.values, 
        columns=['predicted_logS', 'predicted_logS_stdev'],
        index=df.index
    )

    # 4. Merge the predictions with your original names/SMILES
    final_df = pd.concat([df, clean_results], axis=1)

    print("\n--- FINAL RESULTS ---")
    print(final_df)
    
    # Optional: Save to file
    # final_df.to_csv("solubility_results.csv", index=False)"""


    from controller.predictor_runner import process_sample_file
    from controller.predictor_runner import solubility_calculator
    from data_converter.hplc_data_handler import NMR_SOLVENTS
    from data_converter.hplc_data_handler import prepare_fastsolv_input

    """process_sample_file("/Users/arthurbenard/Project 1B/data/Fichier final (RT+sol).xlsx")"""

    """df_orig, df_ready = prepare_fastsolv_input("/Users/arthurbenard/Project 1B/data/Fichier final (RT+sol)_with_smiles.csv", "water")
    final_results = solubility_calculator(df_orig, df_ready,"/Users/arthurbenard/Project 1B/data/Fichier final (RT+sol)_with_smiles.csv")"""

if __name__ == "__main__":
    # The guard is still necessary for the multiprocessing part
    main()