import pandas as pd
import numpy as np
from fastsolv import fastsolv
import os
from data_converter.hplc_data_handler import smiles_by_pubchem_cas, cleaning_array_by_cas
from rdkit import Chem
from rdkit.Chem import Descriptors



# this is to obtain smiles code from given file
def process_sample_file(input_path):
    # 1. Detect file type and load
    file_extension = os.path.splitext(input_path)[1].lower()
    
    print(f"Reading file: {input_path}")

    # Check type of file
    if file_extension == '.xlsx':
        df = pd.read_excel(input_path)
    elif file_extension == '.csv':
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx")
    
    # 2. Check for the necessary columns 'Sample Name' and 'CAS' (specific to what i'm working with)
   
    if 'CAS ' not in df.columns:
        raise KeyError("Could not find a column named 'CAS' in the file.")
    
    # 3. Apply the SMILES function
    print("Fetching SMILES codes... (this may take a moment)")
    df['solute_smiles'] = df['CAS '].apply(smiles_by_pubchem_cas)

    # 4. Clean up: remove rows where SMILES weren't found ("Did not work" text)
    failed_count = (df['solute_smiles'] == "Did not work").sum()

    if failed_count > 0:
        print(f"Warning: Could not find SMILES for {failed_count} samples.")

    # 5. Save as CSV
    output_path = input_path.replace(file_extension, "_with_smiles.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Success! File saved to: {output_path}")

    return df



def solubility_calculator(df_clean, solvent_name, solvent_smiles, temp=298.15):
    
    # 1. Create the exact format FastSolv expects
    fastsolv_ready_df = pd.DataFrame({
        "solute_smiles": df_clean['solute_smiles'].tolist(),
        "solvent_smiles": [solvent_smiles] * len(df_clean),
        "temperature": [temp] * len(df_clean)
    }, index=df_clean.index)

    # 2. Run the engine (fastsolv)
    raw_results = fastsolv(fastsolv_ready_df)

    # 3. Rename columns with the solvent suffix (e.g., logS_methanol)
    results_to_add = pd.DataFrame({
        f'solvent_{solvent_name}': [solvent_name] * len(df_clean),
        f'predicted_logS_{solvent_name}': raw_results['predicted_logS'].values,
        f'predicted_logS_stdev_{solvent_name}': raw_results['predicted_logS_stdev'].values
    }, index=df_clean.index)

    return results_to_add

    

# iteration through solvent dictionary and "stitch" the results onto original data
def solubility_file_matrix(clean_results, solvent_dict, temperature=298.15):
    # Start with the basic info: Names, SMILES, and CAS
    final_df = clean_results[['Sample Name', 'solute_smiles', 'CAS ']].copy()

    # 2. Loop through the solvents
    for solvent_name, solvent_smiles in solvent_dict.items():
        print(f"Predicting solubility in {solvent_name}...")
        
        # Get results for this solvent
        solvent_cols = solubility_calculator(clean_results, solvent_name, solvent_smiles, temperature)
        
        # Merge horizontally
        final_df = pd.concat([final_df, solvent_cols], axis=1)

    # 3. Final Save
    output_name = "data/Master_Solubility_Matrix.xlsx"
    final_df.to_excel(output_name, index=False)
    print(f"Mission Complete! File saved: {output_name}")

    return final_df




def compare_predictions(solubility_matrix, original_data, solvent_dict):
    # Clean the matrix by CAS first to remove redundancy
    clean_matrix_path = cleaning_array_by_cas(solubility_matrix)
    df_results = pd.read_excel(clean_matrix_path)

    # Load original experimental data
    df_original = pd.read_excel(original_data)

    df_results.columns = df_results.columns.str.strip()
    df_original.columns = df_original.columns.str.strip()

    # 2. THE BRIDGE: Merge them together temporarily
    # We only need the 'Sample Name' and the binary columns from the original
    binary_cols = list(solvent_dict.keys())
    cols_to_keep = ['Sample Name', 'RT'] + binary_cols

    # We create a unique lookup table so each Sample Name appears only ONCE
    df_lookup = df_original[cols_to_keep].drop_duplicates(subset=['Sample Name'])
    
    # Now merge with the unique lookup
    df = pd.merge(df_results, df_lookup, on='Sample Name', how='left')


    print(f"Checking coherence for {len(df)} compounds...")

    for solvent_name in solvent_dict.keys():
        # Define our column names based on your new dictionary keys
        logS_col = f'predicted_logS_{solvent_name}'
        stdev_col = f'predicted_logS_stdev_{solvent_name}'
        binary_col = solvent_name  # e.g., 'MeOH', 'ACN'

        if binary_col not in df.columns:
                print(f"Column '{binary_col}' not found in file. Skipping.")
                continue
    # Logical Coherence Check
        # Convert logS to real Solubility (S) in mol/L
        # Using .astype(float) to ensure no errors with 10** power
        df[f'S_{solvent_name}'] = 10 ** df[logS_col].astype(float)
        
        # New Threshold: 0.05 mol/L
        threshold = 0.05

        # Logical Coherence:
        # True if (S > 0.05 AND Exp == 1) OR (S <= 0.05 AND Exp == 0)
        is_coherent = (
            ((df[f'S_{solvent_name}'] > threshold) & (df[binary_col] == 1)) | 
            ((df[f'S_{solvent_name}'] <= threshold) & (df[binary_col] == 0))
        )

        # 4. Create 'Incoherence' column
        # We use .astype(str) so it explicitly says 'True' or 'False'
        coh_col_name = f'Coherence_{solvent_name}'
        df[coh_col_name] = is_coherent.map({True: 'True', False: 'False'})
        
        # 5. Magnitude of difference from the 0.1 threshold
        df[f'Magnitude_{solvent_name}'] = np.where(
            df[coh_col_name] == 'False',
            df[f'S_{solvent_name}'] - threshold,
            np.nan  # Leaves the cell empty if they match
        )

        # 6. Remove StDev to keep it pretty
        if stdev_col in df.columns:
            df.drop(columns=[stdev_col], inplace=True)

    # 7. Final Polish: Save to a new file
    output_file = "data/Compared_Results.xlsx"
    df.to_excel(output_file, index=False)
    
    print(f"Done! Coherence analysis saved to: {output_file}")

    return df         


def calculate_molecular_descriptors(smiles):
    """
    Converts a SMILES string into physical constants: LogP and Molecular Weight.
    Returns (None, None) if the SMILES is invalid.
    """
    
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        return logp, mw
    else:
        return None, None
    

# function that prepares the excel file given at the beginning of the notebook code to prepare it for sklearn
def generate_ml_ready_file_fingerprint_solvent(cleaned_df, solvent_name, solvent_dict, fingerprint_function):

    if solvent_name not in cleaned_df.columns:
        raise ValueError(f"Error: '{solvent_name}' is not a column in the provided DataFrame.")
    
    if solvent_name not in solvent_dict:
        raise ValueError(f"Error: '{solvent_name}' is not in your solvent dictionary.")
    
    # this is a measure to make sure nothing will crash, but normally the file has already been cleaned
    df_filtered = cleaned_df.dropna(subset=['solute_smiles', 'RT', solvent_name]).copy()

    # this is the fingerprinting part
    solvent_smiles = solvent_dict[solvent_name]
    solvent_fp_array = fingerprint_function(solvent_smiles)

    # engineering of the exact df we want, ready for 
    ml_df = pd.DataFrame({
        # Apply the fingerprint function to every solute SMILES, storing the result as an array in the cell
        'Sample Fingerprint': df_filtered['solute_smiles'].apply(fingerprint_function),
                
        # Ensure RT is a float
        'RT': df_filtered['RT'].astype(float),
        
        # Populate every row with the exact same solvent fingerprint array
        'Solvent': [solvent_fp_array for _ in range(len(df_filtered))],
        
        # The binary solubility result (0 or 1)
        'Soluble': df_filtered[solvent_name].astype(int),

        'SMILES': df_filtered['solute_smiles']
    })
    
    # reset indexing step important, in case rows were removed
    ml_df = ml_df.reset_index(drop=True)

    return ml_df


def generate_ml_ready_file_onehot_solvent(cleaned_df, solvent_name, solvent_dict, fingerprint_function):

    if solvent_name not in cleaned_df.columns:
        raise ValueError(f"Error: '{solvent_name}' is not a column in the provided DataFrame.")
    
    if solvent_name not in solvent_dict:
        raise ValueError(f"Error: '{solvent_name}' is not in your solvent dictionary.")
    
    # this is a measure to make sure nothing will crash, but normally the file has already been cleaned
    df_filtered = cleaned_df.dropna(subset=['solute_smiles', 'RT', solvent_name]).copy()

    all_solvents = list(solvent_dict.keys())

    # this creates an array filled with 0, then associates position of the solvent and where to insert a 1, that's why it looks for the index. don't forget it takes i a dictionary
    one_hot_array = np.zeros(len(all_solvents), dtype=int)
    solvent_index = all_solvents.index(solvent_name)
    one_hot_array[solvent_index] = 1
    

    # engineering of the exact df we want, ready for 
    ml_df = pd.DataFrame({
        # Apply the fingerprint function to every solute SMILES, storing the result as an array in the cell
        'Sample Fingerprint': df_filtered['solute_smiles'].apply(fingerprint_function),
                
        # Ensure RT is a float
        'RT': df_filtered['RT'].astype(float),
        
        # Populate every row with the exact same solvent fingerprint array
        'Solvent': [one_hot_array for _ in range(len(df_filtered))],
        
        # The binary solubility result (0 or 1)
        'Soluble': df_filtered[solvent_name].astype(int),

        'SMILES': df_filtered['solute_smiles']
    })
    
    # reset indexing step important, in case rows were removed
    ml_df = ml_df.reset_index(drop=True)

    return ml_df