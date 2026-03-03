from urllib.request import urlopen
from urllib.parse import quote
import pandas as pd
from fastsolv import fastsolv
import os
import pubchempy as pcp
import json


# Dictionary for NMR solvents (i don't know how big this has to be)
NMR_SOLVENTS = {
    "chloroform": "ClC(Cl)Cl",
    "dmso": "CS(=O)C",
    "methanol": "CO",
    "benzene": "c1ccccc1",
    "water": "O"
}

# this function will fetch the SMILES code of a molecule using the CAS number present in the file
# it uses the url written under to efficiently find the code, puts it in a dictionary and extracts the SMILES code 
def smiles_by_pubchem_cas(cas_number):
    cas = str(cas_number).strip() if pd.notna(cas_number) else ""

    if cas:
        try:
            url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/'+ quote(cas) +'/property/SMILES/JSON'
            response = urlopen(url, timeout=5).read().decode('utf8')
            data = json.loads(response)
            
            # Extract the properties list
            props = data['PropertyTable']['Properties'][0]
            
            # Use .get() to look for 'SMILES'
            # This is safer than props['SMILES'] because it won't crash if the key is missing
            smiles = props.get('SMILES')
            return smiles
        except:
            pass
    return 'Did not work'


# after reading the excel sheet with all the solvents, csv file generated and we need to read 
# it in order to feed the correct data to the solubility calculator
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
