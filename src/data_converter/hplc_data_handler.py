from urllib.request import urlopen
from urllib.parse import quote
import pandas as pd
from fastsolv import fastsolv
import os
import pubchempy as pcp
import json


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