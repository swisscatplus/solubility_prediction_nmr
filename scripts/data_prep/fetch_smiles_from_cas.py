import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # make src/ importable
import pandas as pd 
import pubchempy as pcp
import time
import csv
import os
from config import EPFL_INVENTORY, EPFL_INVENTORY_SMILES, EPFL_INVENTORY_SMILES_PROGRESS, ensure_dir

def main():
    # Load the inventory CSV
    df = pd.read_csv(EPFL_INVENTORY, encoding='latin-1')

    # Show the columns
    print(df.columns.tolist())

    # Convert to string and strip surrounding spaces, to avoid false mismatches
    df["CAS #"] = df["CAS #"].astype(str).str.strip()
    # (string format, to compare with PubChem)

    # Keep only rows with an actual CAS (drop NaN and empty cells)
    df_molecules = df[df["CAS #"].notna() & (df["CAS #"] != "") & (df["CAS #"].str.lower() != "nan")].copy()

    print(f"Rows before: {len(df)}")
    print(f"Rows after:  {len(df_molecules)}")

    # Look up each CAS on PubChem
    def get_smiles_from_cas(cas_number):
        """Return the SMILES of a CAS number from PubChem (None if not found)."""
        try:
            compounds = pcp.get_compounds(cas_number, 'name')  # the CAS is searched as a name/identifier
            if compounds:
                return compounds[0].canonical_smiles
            else:
                return None
        except Exception as e:
            print(f"Error for CAS {cas_number} : {e}")
            return None
        
    progress_file = ensure_dir(EPFL_INVENTORY_SMILES_PROGRESS)

    if os.path.exists(progress_file):
        smiles_list = pd.read_csv(progress_file)["SMILES"].tolist()
        print(f"Resuming: {len(smiles_list)} CAS already processed.")
    else:
        smiles_list = []

    start_index = len(smiles_list)  # resume where we stopped

    for i in range(start_index, len(df_molecules)):
        cas = df_molecules["CAS #"].iloc[i]
        smiles = get_smiles_from_cas(cas)
        smiles_list.append(smiles)
        time.sleep(1)  # do not overload the PubChem API

        # Save progress every 100 CAS
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(df_molecules)} processed...")
            pd.DataFrame({"SMILES": smiles_list}).to_csv(progress_file, index=False)

    # Final progress save (in case the last batch of < 100 was not saved)
    pd.DataFrame({"SMILES": smiles_list}).to_csv(progress_file, index=False)

    # Add the SMILES column to the final DataFrame
    df_molecules["SMILES"] = smiles_list

    # Save the complete final file
    output_path = ensure_dir(EPFL_INVENTORY_SMILES)
    df_molecules.to_csv(output_path, index=False)
    print(f"Final file saved: {output_path}")

if __name__ == "__main__":
    main()
