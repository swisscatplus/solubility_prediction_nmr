from urllib.request import urlopen
from urllib.parse import quote
import pandas as pd
from fastsolv import fastsolv
import os


# Dictionary for NMR solvents (i don't know how big this has to be)
NMR_SOLVENTS = {
    "chloroform": "ClC(Cl)Cl",
    "dmso": "CS(=O)C",
    "methanol": "CO",
    "benzene": "c1ccccc1",
    "water": "O"
}


def smiles_code(name):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(name) + '/smiles' #Opens up a page with SMILES code
        ans = urlopen(url).read().decode('utf8') #Reads answer
        return ans
    except:
        return 'Did not work' #If the molecule name is not well written or doesn't exist, it returns nothing
    



#after reading the excel sheet with all the solvents, csv file generated and we need to read 
#it in order to feed the correct data to the solubility calculator
def prepare_fastsolv_input(csv_file, solvent_name, temp=298.15):
# 1. make new data frame using the one obtained with process_sample_file
    df = pd.read_csv(csv_file)

    # 2. Filter: Keep only rows where SMILES is NOT "Did not work"
    # We use .copy() to ensure we have a fresh dataframe for the results
    df_clean = df[df['solute_smiles'] != "Did not work"].copy()
    print(df_clean)

    if df_clean.empty:
        print("No valid SMILES found. Stopping.")
        return None
    
    # 3. Get Solvent SMILES from your dictionary
    solv_smi = NMR_SOLVENTS.get(solvent_name.lower())
    if not solv_smi:
        raise ValueError(f"Solvent '{solvent_name}' not supported.")
    
    # 4. Create the exact format FastSolv expects
    # create a new dataframe with the specific column names
    fastsolv_ready_df = pd.DataFrame({
        "solute_smiles": df_clean['solute_smiles'].tolist(),
        "solvent_smiles": [solv_smi] * len(df_clean),
        "temperature": [temp] * len(df_clean)
    }, index=df_clean.index) # Keep the original index!

    # Add solvent_name to the clean df for the final display/file
    df_clean['solvent_name'] = solvent_name

    return df_clean, fastsolv_ready_df
