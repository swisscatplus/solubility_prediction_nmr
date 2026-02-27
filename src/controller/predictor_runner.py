import pandas as pd
import numpy as np
from fastsolv import fastsolv
import os
from data_converter.hplc_data_handler import smiles_code



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
    
    # 2. Check for the 'Sample Name' column (specific to what i'm working with)
    if 'Sample Name' not in df.columns:
        raise KeyError("Could not find a column named 'Sample Name' in the file.")
    
    # 3. Apply the SMILES function
    print("Fetching SMILES codes from NIH Cactus... (this may take a moment)")
    # 'apply' runs function on every row
    df['solute_smiles'] = df['Sample Name'].apply(smiles_code)

    # 4. Clean up: remove rows where SMILES weren't found ("Did not work" text)
    failed_mask = df['solute_smiles'] == "Did not work"
    failed_count = failed_mask.sum()

    if failed_count > 0:
        print(f"Warning: Could not find SMILES for {failed_count} samples.")

    # 5. Save as CSV
    output_path = input_path.replace(file_extension, "_with_smiles.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Success! File saved to: {output_path}")

    
    return df


def solubility_calculator(df_original, df_input, original_filename):
    
    # Runs the calculation and merges results back with the original names.
    print(f"Running FastSolv on {len(df_input)} samples...")

    # 1. Perform calculation with fastsolv
    raw_results = fastsolv(df_input)

    # 2. Format results to match df_input index
    clean_results = pd.DataFrame(
        raw_results.values, 
        columns=['predicted_logS', 'predicted_logS_stdev'],
        index=df_input.index
    )

    # 3. Merge with original context (names, etc.), to make one big dataframe with everything
    final_df = pd.concat([df_original, clean_results], axis=1)


    # 4. THE CLEANUP: Keep only the 6 essential columns
    # This ensures 'useless' columns are removed from the final CSV
    desired_columns = [
        'Sample Name', 'solvent_name', 'temperature', 
        'solute_smiles', 'predicted_logS', 'predicted_logS_stdev'
    ]
    
    # Filter final_df to only include columns that actually exist
    final_df = final_df[[col for col in desired_columns if col in final_df.columns]]


    # 5. Save file
    name_part, ext = os.path.splitext(original_filename)
    output_name = f"{name_part}_results{ext}"
    final_df.to_csv(output_name, index=False)

   # 6. Display to console
    print("-" * 60)
    # We round to 3 for the display to keep the terminal tidy
    print(final_df.round(3).to_string(index=False)) 
    print("-" * 60)
    print(f"Cleaned data saved to: {output_name}\n")

    
    return final_df

