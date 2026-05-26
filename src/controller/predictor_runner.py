import pandas as pd
import numpy as np
from fastsolv import fastsolv
import os
from data_converter.hplc_data_handler import smiles_by_pubchem_cas, cleaning_array_by_cas
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
import xgboost as xgb


# This is to obtain smiles code from given file
def process_sample_file(input_path):
    file_extension = os.path.splitext(input_path)[1].lower()
    
    print(f"Reading file: {input_path}")

    # Check type of file
    if file_extension == '.xlsx':
        df = pd.read_excel(input_path)
    elif file_extension == '.csv':
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx")
    
    # Check for the necessary columns 'Sample Name' and 'CAS' (specific to what i'm working with)
   
    if 'CAS ' not in df.columns:
        raise KeyError("Could not find a column named 'CAS' in the file.")
    
    # Apply the SMILES function
    print("Fetching SMILES codes... (this may take a moment)")
    df['solute_smiles'] = df['CAS '].apply(smiles_by_pubchem_cas)

    # Clean up: remove rows where SMILES weren't found ("Did not work" text)
    failed_count = (df['solute_smiles'] == "Did not work").sum()

    if failed_count > 0:
        print(f"Warning: Could not find SMILES for {failed_count} samples.")

    # Save as CSV
    output_path = input_path.replace(file_extension, "_with_smiles.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Success! File saved to: {output_path}")

    return df



def solubility_calculator(df_clean, solvent_name, solvent_smiles, temp=298.15):
    
    # Create the exact format FastSolv expects
    fastsolv_ready_df = pd.DataFrame({
        "solute_smiles": df_clean['solute_smiles'].tolist(),
        "solvent_smiles": [solvent_smiles] * len(df_clean),
        "temperature": [temp] * len(df_clean)
    }, index=df_clean.index)

    # Run the engine (fastsolv)
    raw_results = fastsolv(fastsolv_ready_df)

    # Rename columns with the solvent suffix (e.g., logS_methanol)
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

    # Loop through the solvents
    for solvent_name, solvent_smiles in solvent_dict.items():
        print(f"Predicting solubility in {solvent_name}...")
        
        # Get results for this solvent
        solvent_cols = solubility_calculator(clean_results, solvent_name, solvent_smiles, temperature)
        
        # Merge horizontally
        final_df = pd.concat([final_df, solvent_cols], axis=1)

    output_name = "/Users/arthurbenard/Project 1B/data/Master_Solubility_Matrix.xlsx"
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

    # Merge them together temporarily
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

        # Create 'Incoherence' column
        # We use .astype(str) so it explicitly says 'True' or 'False'
        coh_col_name = f'Coherence_{solvent_name}'
        df[coh_col_name] = is_coherent.map({True: 'True', False: 'False'})
        
        # Magnitude of difference from the 0.1 threshold
        df[f'Magnitude_{solvent_name}'] = np.where(
            df[coh_col_name] == 'False',
            df[f'S_{solvent_name}'] - threshold,
            np.nan  # Leaves the cell empty if they match
        )

        # Remove StDev to keep it pretty
        if stdev_col in df.columns:
            df.drop(columns=[stdev_col], inplace=True)

    # Final Polish: Save to a new file
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
    

# Function that prepares the excel file given at the beginning of the notebook code to prepare it for sklearn
def generate_ml_ready_file_fingerprint_solvent(cleaned_df, solvent_name, solvent_dict, fingerprint_function):

    if solvent_name not in cleaned_df.columns:
        raise ValueError(f"Error: '{solvent_name}' is not a column in the provided DataFrame.")
    
    if solvent_name not in solvent_dict:
        raise ValueError(f"Error: '{solvent_name}' is not in your solvent dictionary.")
    
    # This is a measure to make sure nothing will crash, but normally the file has already been cleaned
    df_filtered = cleaned_df.dropna(subset=['solute_smiles', 'RT', solvent_name]).copy()

    # This is the fingerprinting part
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
    
    # Reset indexing step important, in case rows were removed
    ml_df = ml_df.reset_index(drop=True)

    return ml_df


def generate_ml_ready_file_onehot_solvent(cleaned_df, solvent_name, solvent_dict, fingerprint_function):

    if solvent_name not in cleaned_df.columns:
        raise ValueError(f"Error: '{solvent_name}' is not a column in the provided DataFrame.")
    
    if solvent_name not in solvent_dict:
        raise ValueError(f"Error: '{solvent_name}' is not in your solvent dictionary.")
    
    # This is a measure to make sure nothing will crash, but normally the file has already been cleaned
    df_filtered = cleaned_df.dropna(subset=['solute_smiles', 'RT', solvent_name]).copy()

    all_solvents = list(solvent_dict.keys())

    # One-hot Coding: this creates an array filled with 0, then associates position of the solvent and where to insert a 1, that's why it looks for the index. don't forget it takes i a dictionary
    one_hot_array = np.zeros(len(all_solvents), dtype=int)
    solvent_index = all_solvents.index(solvent_name)
    one_hot_array[solvent_index] = 1
    

    # Engineering of the exact df we want, ready for 
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
    
    # Reset indexing step important, in case rows were removed
    ml_df = ml_df.reset_index(drop=True)

    return ml_df


# This next function works just like the one above but it adds the "logS" values for ONE SOLVENT, one more parameter to train on
def generate_ml_file_with_logs(cleaned_df, solvent_name, solvent_dict, fingerprint_function):

    logs_col = f"predicted_logS_{solvent_name}"
    
    # General checks
    if solvent_name not in cleaned_df.columns:
        raise ValueError(f"Missing binary column: '{solvent_name}'")
    if logs_col not in cleaned_df.columns:
        raise ValueError(f"Missing logS column: '{logs_col}'")


    # This is the same measure taken before to make sure nothing will crash but this time we're making sure there's values in the logS column
    # but normally the file has already been cleaned beforehand
    df_filtered = cleaned_df.dropna(subset=['solute_smiles', 'RT', solvent_name, logs_col]).copy()

    # One-hot coding
    all_solvents = list(solvent_dict.keys())
    one_hot_array = np.zeros(len(all_solvents), dtype=int)
    solvent_index = all_solvents.index(solvent_name)
    one_hot_array[solvent_index] = 1


    # df construction: adding a new column
    ml_df = pd.DataFrame({
        'Sample Fingerprint': df_filtered['solute_smiles'].apply(fingerprint_function),
        'RT': df_filtered['RT'].astype(float),
  
        'logS': df_filtered[logs_col].astype(float),
        
        'Solvent': [one_hot_array for _ in range(len(df_filtered))],
        'Soluble': df_filtered[solvent_name].astype(int),
        'SMILES': df_filtered['solute_smiles']
    })
    
    ml_df = ml_df.reset_index(drop=True)

    return ml_df



def add_logs_to_df(cleaned_df, sol_matrix,solvent_dict):
    # to add the logs values
    exact_log_cols = [f"predicted_logS_{sol}" for sol in solvent_dict.keys()]
    valid_log_cols = [col for col in exact_log_cols if col in sol_matrix.columns]
    if not valid_log_cols:
        raise ValueError("Error: Could not find any of your specified predicted_logS columns")
    
    columns_to_keep = ['CAS '] + valid_log_cols
    logs_only_df = sol_matrix[columns_to_keep].drop_duplicates(subset=['CAS '])

    merged_df = pd.merge(cleaned_df, logs_only_df, on='CAS ', how='left')
    
    print(f"Successfully added {len(valid_log_cols)} exact logS columns")
    return merged_df



def add_matrix_id_to_df(mega_df, raw_df, matrix_col_name, smiles_col='solute_smiles'):

    print(f"One-Hot Encoding the '{matrix_col_name}' column...")
    
    # Grab just the SMILES and your matrix column
    temp_data = raw_df[[smiles_col, matrix_col_name]]
    
    # Merge it into the master dataframe
    new_df = mega_df.merge(temp_data, on=smiles_col, how='left')
    
    # Fill any missing rows with 'Unknown' so the math doesn't break
    new_df[matrix_col_name] = new_df[matrix_col_name].fillna('Unknown')
    
    # The One-Hot Encoder
    encoder = OneHotEncoder(sparse_output=False, dtype=int)
    encoded_grid = encoder.fit_transform(new_df[[matrix_col_name]])
    
    # Pack the grid into lists inside a single new column
    new_df['Matrix_Packed_Array'] = list(encoded_grid)
    
    # Drop the original text column
    new_df = new_df.drop(columns=[matrix_col_name])

    categories = encoder.categories_[0]
    print(f"   Successfully packed {len(categories)} matrices: {categories}")
    print(f"   -> Example array looks like: {new_df['Matrix_Packed_Array'].iloc[0]}")
    
    return new_df



def run_descriptor_competition(df_desc, descriptors):
    """
    Takes a dataframe with calculated RDKit descriptors, splits it securely, 
    scales the continuous variables, and runs a feature tournament.
    """
    print(" Preparing Data, Splitting, and Scaling (Leakage-Free)...")
    
    # Grab all the raw arrays
    X_fp_raw = np.vstack(df_desc['Sample Fingerprint'].values)
    X_solvent = np.vstack(df_desc['Solvent'].values)
    X_matrix = np.vstack(df_desc['Matrix_Packed_Array'].values)
    
    # Grab the new descriptors as a 2D array
    X_desc = df_desc[['MolLogP', 'TPSA', 'MolWt', 'NumHDonors', 'NumHAcceptors']].values
    
    y = df_desc['Soluble'].values.astype(int)
    groups = df_desc['SMILES'].values
    
    # SECURE SPLIT FIRST
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=34)
    train_idx, test_idx = next(gss.split(X_fp_raw, y, groups))
    
    X_fp_train, X_fp_test = X_fp_raw[train_idx], X_fp_raw[test_idx]
    X_sol_train, X_sol_test = X_solvent[train_idx], X_solvent[test_idx]
    X_mat_train, X_mat_test = X_matrix[train_idx], X_matrix[test_idx]
    X_desc_train, X_desc_test = X_desc[train_idx], X_desc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # DIMENSION REDUCTION 
    k_best = SelectKBest(score_func=chi2, k=descriptors)
    X_fp_train_best = k_best.fit_transform(X_fp_train, y_train)
    X_fp_test_best = k_best.transform(X_fp_test)
    
    # SCALING DESCRIPTORS 
    scaler = StandardScaler()
    X_desc_train_scaled = scaler.fit_transform(X_desc_train)
    X_desc_test_scaled = scaler.transform(X_desc_test)
    
    # Slice and Reshape into 1D columns for the tournament
    X_logp_train = X_desc_train_scaled[:, 0].reshape(-1, 1)
    X_logp_test = X_desc_test_scaled[:, 0].reshape(-1, 1)
    
    X_tpsa_train = X_desc_train_scaled[:, 1].reshape(-1, 1)
    X_tpsa_test = X_desc_test_scaled[:, 1].reshape(-1, 1)
    
    X_molwt_train = X_desc_train_scaled[:, 2].reshape(-1, 1)
    X_molwt_test = X_desc_test_scaled[:, 2].reshape(-1, 1)

    X_hdon_train = X_desc_train_scaled[:, 3].reshape(-1, 1)
    X_hdon_test = X_desc_test_scaled[:, 3].reshape(-1, 1)

    X_hacc_train = X_desc_train_scaled[:, 4].reshape(-1, 1)
    X_hacc_test = X_desc_test_scaled[:, 4].reshape(-1, 1)

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f" -> Calculated scale_pos_weight: {scale_weight:.2f}")

    print(" Running the Descriptor Tournament...\n")

    # Pre-glue the baseline 
    X_base_train = np.hstack((X_fp_train_best, X_sol_train, X_mat_train))
    X_base_test = np.hstack((X_fp_test_best, X_sol_test, X_mat_test))

    # Define the 5 experimental setups
    experiments = {
        "1. Baseline (No Descriptors)": (X_base_train, X_base_test),
        "2. Baseline + MolLogP":        (np.hstack((X_base_train, X_logp_train)), np.hstack((X_base_test, X_logp_test))),
        "3. Baseline + TPSA":           (np.hstack((X_base_train, X_tpsa_train)), np.hstack((X_base_test, X_tpsa_test))),
        "4. Baseline + MolWt":          (np.hstack((X_base_train, X_molwt_train)), np.hstack((X_base_test, X_molwt_test))),
        "5. Baseline + LogP + TPSA":    (np.hstack((X_base_train, X_logp_train, X_tpsa_train)), np.hstack((X_base_test, X_logp_test, X_tpsa_test))),
        "6. Baseline + LogP + MolWt":   (np.hstack((X_base_train, X_logp_train, X_molwt_train)), np.hstack((X_base_test, X_logp_test, X_molwt_test))),
        "7. Baseline + TPSA + MolWt":   (np.hstack((X_base_train, X_tpsa_train, X_molwt_train)), np.hstack((X_base_test, X_tpsa_test, X_molwt_test))),
        "8. Baseline + H-Bonds (Donors & Acceptors)": (np.hstack((X_base_train, X_hdon_train, X_hacc_train)), np.hstack((X_base_test, X_hdon_test, X_hacc_test))),
        "9. Baseline + TPSA + H-Bonds (Donors & Acceptors)": (np.hstack((X_base_train, X_tpsa_train, X_hdon_train, X_hacc_train)), np.hstack((X_base_test, X_tpsa_test, X_hdon_test, X_hacc_test))),
        "10. All 5":  (np.hstack((X_base_train, X_desc_train_scaled)), np.hstack((X_base_test, X_desc_test_scaled)))
    }

    results = []

    for name, (X_tr, X_te) in experiments.items():
        # Train the model
        model = xgb.XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=6, 
            random_state=69, 
            eval_metric='logloss',
            scale_pos_weight=scale_weight, 
            n_jobs=-1
        )
        model.fit(X_tr, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        
        results.append({"Experiment": name, "Accuracy": acc, "Precision": prec})

    # Format the output into a nice pandas DataFrame
    results_df = pd.DataFrame(results)

    print("==================================================")
    print("             TOURNAMENT RESULTS                   ")
    print("==================================================")
    print(results_df.to_string(index=False))
    print("==================================================")
    
    return results_df


def run_hyperparameter_competition(df_desc):
    """
    Isolates the Champion Dataset (Baseline + TPSA + MolWt), prevents leakage, 
    calculates class weights, and runs a GridSearch to find the best XGBoost hyperparameters 
    optimized for precision.
    """
    print(" Locking in 'Baseline + TPSA + MolWt' Dataset...")

    # Grab raw arrays
    X_fp_raw = np.vstack(df_desc['Sample Fingerprint'].values)
    X_sol = np.vstack(df_desc['Solvent'].values)
    X_mat = np.vstack(df_desc['Matrix_Packed_Array'].values)
    X_desc = df_desc[['TPSA', 'MolWt']].values # Only extracting our two champion descriptors

    y = df_desc['Soluble'].values.astype(int)
    groups = df_desc['SMILES'].values

    # SECURE SPLIT
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=34)
    train_idx, test_idx = next(gss.split(X_fp_raw, y, groups))

    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx] # Crucial for the GridSearch later!

    # Calculate class weight
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

    # PROCESS FEATURES (Strictly on Train to prevent leakage)
    k_best = SelectKBest(score_func=chi2, k=200)
    X_fp_train_best = k_best.fit_transform(X_fp_raw[train_idx], y_train)
    X_fp_test_best = k_best.transform(X_fp_raw[test_idx])

    scaler = StandardScaler()
    X_desc_train_scaled = scaler.fit_transform(X_desc[train_idx])
    X_desc_test_scaled = scaler.transform(X_desc[test_idx])

    # Glue the Champion datasets together
    X_train_champ = np.hstack((X_fp_train_best, X_sol[train_idx], X_mat[train_idx], X_desc_train_scaled))
    X_test_champ = np.hstack((X_fp_test_best, X_sol[test_idx], X_mat[test_idx], X_desc_test_scaled))

    print(" Initializing Leak-Free Grid Search (Optimizing strictly for Precision)...")

    # Force GridSearch to respect molecule boundaries
    gkf = GroupKFold(n_splits=3) 

    # The Base Engine
    xgb_tuner = xgb.XGBClassifier(
        random_state=69, 
        eval_metric='logloss',
        scale_pos_weight=scale_weight,
        n_jobs=-1
    )

    # The Dials we want to tweak
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250],         # How many trees to build
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.25, 0.3], # How aggressively it corrects mistakes
        'max_depth': [4, 5, 6, 7, 8, 9],             # How deep the logic questions go
        'subsample': [0.8, 1.0],             # Uses a random % of data per tree to prevent memorization
        'gamma': [0, 0.1, 1, 5],                  # Prunes weak logic trees
        'min_child_weight': [1, 3, 5, 7],         # Forces rules to apply to groups, not single outliers
        'reg_alpha': [0, 0.1, 1, 10],             # L1 Regularization: zero-outs useless fingerprint bits
        'reg_lambda': [1, 1.5, 5, 10]             # L2 Regularization: shrinks over-dominant features
    }

    # The Tuner
    random_search = RandomizedSearchCV(
        estimator=xgb_tuner,
        param_distributions=param_grid,
        n_iter=100,           # Test 100 random combinations from the massive grid
        scoring='precision',  # Telling it to prioritize false-positive reduction
        cv=gkf,
        verbose=1,
        random_state=87,
        n_jobs=-1
    )

    # Run the tournament of hyperparameters
    random_search.fit(X_train_champ, y_train, groups=groups_train)

   
    # RESULTS
    best_model = random_search.best_estimator_

    print("\n==================================================")
    print("              BEST HYPERPARAMETERS                ")
    print("==================================================")
    for key, value in random_search.best_params_.items():
        print(f"{key}: {value}")

    # Test the ultimate tuned model on the untouched test set
    y_pred_tuned = best_model.predict(X_test_champ)
    final_acc = accuracy_score(y_test, y_pred_tuned)
    final_prec = precision_score(y_test, y_pred_tuned, zero_division=0)

    print("\n==================================================")
    print("            TUNED CHAMPION RESULTS                ")
    print("==================================================")
    print(f"Final Accuracy:  {final_acc:.4f}")
    print(f"Final Precision: {final_prec:.4f}")
    print("==================================================")
    
    return best_model, random_search.best_params_


def unknown_column_machine_test(df, target_machine):
    print(f"Running Test on Unseen Machine: {target_machine}\n")
    

    # Creating a mask to find ONLY the rows where the machine matches our target
    # .copy() ensures we don't accidentally modify our master dataframe, which would be a shame
    df_test = df[df['Matrix ID Name'] == target_machine].copy()
    
    # We keep all rows where the machine is NOT our target
    df_train = df[df['Matrix ID Name'] != target_machine].copy()
    
    # Resetting the index just cleans up the row numbers so Pandas doesn't get confused later
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    print(f"Training Rows (6 Machines, No Leaks): {len(df_train)}")
    print(f"Testing Rows  (1 Machine):  {len(df_test)}")

    # Same thing as we usually do for defining model variables
    y_train = df_train['Soluble'].values
    
    X_rt_train = df_train[['RT']].values 
    X_phys_train = df_train[['MolWt', 'Sol_Dielectric', 'Sol_Hansen_D', 'Sol_Hansen_P', 'Sol_Hansen_H']].values
    
    X_mat_train = np.vstack(df_train['Matrix_Packed_Array'].values)
    
    # [Retention Time] + [Matrix Vectors] and the other available data
    X_train_final = np.hstack((X_rt_train, X_phys_train, X_mat_train))
    
    # We do the exact same extraction process, but exclusively for our isolated test dataframe
    y_test = df_test['Soluble'].values
    X_rt_test = df_test[['RT']].values 
    X_phys_test = df_test[['MolWt', 'Sol_Dielectric', 'Sol_Hansen_D', 'Sol_Hansen_P', 'Sol_Hansen_H']].values
    X_mat_test = np.vstack(df_test['Matrix_Packed_Array'].values)
    
    X_test_final = np.hstack((X_rt_test, X_phys_test, X_mat_test))
    
    scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    model = xgb.XGBClassifier(n_estimators=50, learning_rate=0.05, max_depth=4, subsample=0.8,
                                  colsample_bytree=1.0, gamma=1, min_child_weight=5, reg_lambda=10, 
                                  scale_pos_weight=scale_weight, random_state=72, n_jobs=-1)


    # The AI learns the rules of chemistry using ONLY the 6 known machines
    model.fit(X_train_final, y_train)
    
    # The AI attempts to predict solubility on the 1 machine it has never seen
    y_pred = model.predict(X_test_final)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    
    print("\n Experiment results:")
    print(f"Accuracy on completely unseen {target_machine}:  {acc:.4f}")
    print(f"Precision on completely unseen {target_machine}: {prec:.4f}")
    print("-" * 50)
    
    return model, df_train, df_test



def strict_unknown_column_machine_test(df, target_machine, test_size=0.3):
    print(f"Running STRICT on Unknown Machine: {target_machine}\n")
    
    # We group by 'SMILES' so a molecule and all of its machine runs go 100% to Train OR 100% to Test
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=89)
    
    # Generate the row indices for our split, "groups=df['SMILES']" means do not separate twins
    # gss.split(...) starts tossing them into a Train bucket and a Test bucket until it hits 70/30 split ratio
    # next(...) extracts that split, it doesn't do it without it
    train_idx, test_idx = next(gss.split(df, groups=df['SMILES']))
    
    # Create two completely isolated pools of molecules
    df_train_pool = df.iloc[train_idx].copy()
    df_test_pool = df.iloc[test_idx].copy()
    
    # Training Set: the "Train Pool" molecules, but ONLY on the 6 known machines
    df_train = df_train_pool[df_train_pool['Matrix ID Name'] != target_machine].copy()
    
    # Testing Set: the "Test Pool" molecules, but ONLY on the 1 unknown machine
    df_test = df_test_pool[df_test_pool['Matrix ID Name'] == target_machine].copy()

    df_test_other_matrices = df_test_pool[df_test_pool['Matrix ID Name'] != target_machine].copy()
    
    # Reset indices to keep Pandas happy and avoid future problems
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_test_other_matrices = df_test_other_matrices.reset_index(drop=True)
    
    print(f"Training Rows (Known Molecules on Known Machines):   {len(df_train)}")
    print(f"Testing Rows  (Unknown Molecules on Unseen Machine): {len(df_test)}")
    print(f"Testing Rows Differently (Unknown Molecules on Known Machines): {len(df_test_other_matrices)}")
    


    # The same old thing (and dance lol)
    y_train = df_train['Soluble'].values
    X_rt_train = df_train[['RT']].values 
    X_phys_train = df_train[['MolWt', 'Sol_Dielectric', 'Sol_Hansen_D', 'Sol_Hansen_P', 'Sol_Hansen_H']].values
    X_mat_train = np.vstack(df_train['Matrix_Packed_Array'].values)
    
    X_train_final = np.hstack((X_rt_train, X_phys_train, X_mat_train))
    
    # Testing features like defined above
    y_test = df_test['Soluble'].values
    X_rt_test = df_test[['RT']].values 
    X_phys_test = df_test[['MolWt', 'Sol_Dielectric', 'Sol_Hansen_D', 'Sol_Hansen_P', 'Sol_Hansen_H']].values
    X_mat_test = np.vstack(df_test['Matrix_Packed_Array'].values)
    
    X_test_final = np.hstack((X_rt_test, X_phys_test, X_mat_test))


    # Testing features, unknown molecules with the other 6 matrices (to compare)
    y_test_other_matrices = df_test_other_matrices['Soluble'].values
    X_rt_test_other_matrices = df_test_other_matrices[['RT']].values 
    X_phys_test_other_matrices = df_test_other_matrices[['MolWt', 'Sol_Dielectric', 'Sol_Hansen_D', 'Sol_Hansen_P', 'Sol_Hansen_H']].values
    X_mat_test_other_matrices = np.vstack(df_test_other_matrices['Matrix_Packed_Array'].values)
    
    X_test_final_other_matrices = np.hstack((X_rt_test_other_matrices, X_phys_test_other_matrices, X_mat_test_other_matrices))
    
    # Training models
    scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    model = xgb.XGBClassifier(n_estimators=50, learning_rate=0.05, max_depth=4, subsample=0.8,
                                  colsample_bytree=1.0, gamma=1, min_child_weight=5, reg_lambda=10, 
                                  scale_pos_weight=scale_weight, random_state=69, n_jobs=-1)
    
    model.fit(X_train_final, y_train)
    

    # Scoring
    y_pred = model.predict(X_test_final)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)

    y_pred_other_matrices = model.predict(X_test_final_other_matrices)
    acc_2 = accuracy_score(y_test_other_matrices, y_pred_other_matrices)
    prec_2 = precision_score(y_test_other_matrices, y_pred_other_matrices, zero_division=0)
    
    print("\nSTRICT TESTING RESULTS:")
    print(f"Accuracy (Unseen Molecule + Unseen Machine):  {acc:.4f}")
    print(f"Precision (Unseen Molecule + Unseen Machine): {prec:.4f}")
    print("-" * 50)
    print(f"Accuracy (Unseen Molecule + Seen Machines):  {acc_2:.4f}")
    print(f"Precision (Unseen Molecule + Seen Machines): {prec_2:.4f}")
    print("-" * 50)
    
    return acc, acc_2