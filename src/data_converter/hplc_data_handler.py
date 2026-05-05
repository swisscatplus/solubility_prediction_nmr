from urllib.request import urlopen
from urllib.parse import quote
import pandas as pd
from fastsolv import fastsolv
import os
import pubchempy as pcp
import json
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, Lipinski
import numpy as np
from sklearn.decomposition import PCA

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




def scrub(df, smiles_col='SMILES'):
    """
    Tests every SMILES string with RDKit. 
    Drops rows that are blank or chemically invalid.
    """
    print(f"🧹 Starting cleanup. Original rows: {len(df)}")
    
    # 1. Drop obvious blank cells first
    df_clean = df.dropna(subset=[smiles_col]).copy()
    
    # 2. The RDKit Acid Test: Try to build a molecule from each string
    # We use str(x) just in case a number accidentally sneaked into the column!
    valid_mask = df_clean[smiles_col].apply(lambda x: Chem.MolFromSmiles(str(x)) is not None)
    
    # 3. Keep only the ones that passed
    df_valid = df_clean[valid_mask].copy()
    
    # Report the casualties
    dropped = len(df) - len(df_valid)
    print(f"   -> Dropped {dropped} rows containing invalid or missing SMILES.")
    print(f"✅ Cleaned row count: {len(df_valid)}")
    
    return df_valid


# This function is to compress the 2000 something bits of the morgan fingerprints, using PCA.
def compress_fingerprints_pca(df, fp_col_name, variance_to_keep=0.95):
   
    print(f"Compressing '{fp_col_name}' using PCA...")
    print(f"-> Target: Keep {variance_to_keep * 100}% of the original chemical information.")
    
    new_df = df.copy()
    
    # 1. Extract and stack the fingerprints into a massive 2D matrix
    # (np.vstack stacks all the individual lists on top of each other so PCA can read them)
    fps_matrix = np.vstack(new_df[fp_col_name].values)
    original_dims = fps_matrix.shape[1]
    
    # 2. Build and run the PCA compressor
    # By passing a float, we tell PCA to figure out exactly how many 
    # columns it needs to keep 95% of the variance.
    pca = PCA(n_components=variance_to_keep, random_state=34)
    compressed_matrix = pca.fit_transform(fps_matrix)
    new_dims = compressed_matrix.shape[1]
    
    # 3. Pack the shiny new, dense arrays back into the dataframe, overwriting the old ones
    new_df[fp_col_name] = list(compressed_matrix)
    
    print(f"   Compression Complete!")
    print(f"   -> Original Dimensions: {original_dims} bits")
    print(f"   -> New Dimensions:      {new_dims} dense features")
    
    return new_df


def calculate_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            molwt = Descriptors.MolWt(mol)
            h_donors = Lipinski.NumHDonors(mol)
            h_acceptors = Lipinski.NumHAcceptors(mol)
            return pd.Series([logp, tpsa, molwt, h_donors, h_acceptors])
        else:
            return pd.Series([0.0, 0.0, 0.0, 0.0, 0.0]) # Fallback for invalid SMILES
    except:
        return pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])



def calculate_mlwt(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            molwt = Descriptors.MolWt(mol)
            h_donors = Lipinski.NumHDonors(mol)
            h_acceptors = Lipinski.NumHAcceptors(mol)
            return pd.Series([ molwt, h_donors, h_acceptors])
        else:
            return pd.Series([0.0, 0.0, 0.0]) # Fallback for invalid SMILES
    except:
        return pd.Series([0.0, 0.0, 0.0])



# A dictionary holding the physical properties for specific solvents.
# Hansen invented these parameters to map out solvents in a 3D grid based on three specific types of energy: Dispersion, Polarity, H-bonding
solvent_physics_db = {
    'MeOH': {'Dielectric': 32.7, 'Hansen_D': 15.1, 'Hansen_P': 12.3, 'Hansen_H': 22.3},
    'ACN': {'Dielectric': 37.5, 'Hansen_D': 15.3, 'Hansen_P': 18.0, 'Hansen_H': 6.1},
    'DMSO': {'Dielectric': 46.7, 'Hansen_D': 18.4, 'Hansen_P': 16.4, 'Hansen_H': 10.2},
    'DCM': {'Dielectric': 8.93, 'Hansen_D': 18.2, 'Hansen_P': 6.3, 'Hansen_H': 6.1},
    'CHCl3': {'Dielectric': 4.81, 'Hansen_D': 17.8, 'Hansen_P': 3.1, 'Hansen_H': 5.7}
}


def extract_solvent_physics(solvent_name):
    """Looks up the solvent name and returns its physical properties."""
    # If the solvent isn't in the dictionary, return a safe default (avoid crash)
    props = solvent_physics_db.get(solvent_name, {'Dielectric': 0.0, 'Hansen_D': 0.0, 'Hansen_P': 0.0, 'Hansen_H': 0.0})
    return pd.Series([props['Dielectric'], props['Hansen_D'], props['Hansen_P'], props['Hansen_H']])