from urllib.request import urlopen
from urllib.parse import quote
import pandas as pd
from fastsolv import fastsolv
import os
import pubchempy as pcp
import json
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np

# Dictionary for NMR solvents (i don't know how big this has to be)
NMR_SOLVENTS = {
    "MeOH": "CO",
    "ACN": "CC#N",
    "DMSO": "CS(=O)C",
    "DCM": "C(Cl)Cl",
    "CHCl3": "ClC(Cl)Cl"
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
def prepare_fastsolv_input(df_with_smiles):

    df_clean = df_with_smiles[df_with_smiles['solute_smiles'] != "Did not work"].copy()

    if df_clean.empty:
        print("No valid SMILES found in the file. Stopping.")
        return None
    
    print(f"Data cleaned. {len(df_clean)} compounds ready for prediction.")

    return df_clean

# Cleaning function, because data was repeating i didn't notice all the different analytical columns used 
# Using the cas number to do this, i'm also adding a line of code that removes rows where no signal was obtained
def cleaning_array_by_cas(input_file, cas_column_name='CAS '):
   
    # 1. Load the "dirty" data
    df = pd.read_excel(input_file)
    initial_count = len(df)
    
   # Safety Check: Does the CAS column actually exist?
    if cas_column_name not in df.columns:
        print(f"Error: Could not find a column named '{cas_column_name}'.")
        print(f"Available columns are: {df.columns.tolist()}")
        return None

    # 2. Remove rows where CAS is missing
    df = df.dropna(subset=[cas_column_name])
    
    # 3. Drop Duplicates based ONLY on the CAS number
    # 'keep=first' ensures we don't lose the molecule entirely
    df = df.drop_duplicates(subset=[cas_column_name], keep='first')
    temp_count = len(df)
    removed = initial_count - temp_count

    df_unique = df[df['Signal'] != 0]
    final_count = len(df_unique)
    nothing = initial_count - final_count

    
    print(f"Original Row Count: {initial_count}")
    print(f"Unique Molecules (by CAS): {final_count}")
    print(f"Redundant Rows Deleted: {removed}")
    print(f"No Signal Rows Deleted: {nothing}")
    print("----------------------------------")

    # 4. Save the lean version
    output_file = input_file.replace(".xlsx", "_Unique_CAS.xlsx")
    df_unique.to_excel(output_file, index=False)
    
    return df_unique



def smiles_to_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(2048) # Return empty barcode if SMILES is broken
        
        # Create a 2048-bit Morgan Fingerprint (radius 2 is standard)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return np.array(fp)
    except:
        return np.zeros(2048)

