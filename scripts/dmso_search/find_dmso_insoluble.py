import pandas as pd
import sys
import os
import xgboost as xgb
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # make src/ importable

from config import EPFL_INVENTORY, EPFL_INVENTORY_SMILES_PROGRESS, DMSO_SEARCH, ensure_dir
from controller.predictor_runner import solubility_calculator

# Reload the source file to get the CAS column back
df_source = pd.read_csv(EPFL_INVENTORY, encoding='latin-1')
df_source["CAS #"] = df_source["CAS #"].astype(str).str.strip()
df_molecules_source = df_source[
    df_source["CAS #"].notna() & (df_source["CAS #"] != "") & (df_source["CAS #"].str.lower() != "nan")
].copy()

# Load the SMILES already computed (saved progress)
smiles_temp = pd.read_csv(EPFL_INVENTORY_SMILES_PROGRESS)

# Re-attach CAS and SMILES by position (same processing order)
df_clean = df_molecules_source.iloc[:len(smiles_temp)].copy()
df_clean["SMILES"] = smiles_temp["SMILES"].values

# Rename to match what solubility_calculator expects
df_clean = df_clean.rename(columns={"SMILES": "solute_smiles"})

# Drop rows whose SMILES was not found
df_clean = df_clean[df_clean["solute_smiles"].notna()].copy()
print(f"Molecules with a valid SMILES: {len(df_clean)}")

# Predict logS in DMSO
dmso_smiles = "CS(=O)C"
results_dmso = solubility_calculator(df_clean, solvent_name="DMSO", solvent_smiles=dmso_smiles)
df_clean = df_clean.join(results_dmso)

top50 = df_clean.sort_values("predicted_logS_DMSO", ascending=True).head(50)
top50.to_excel(ensure_dir(DMSO_SEARCH / "top50_dmso_insoluble.xlsx"), index=False)

# --- Draw the structures with RDKit ---
output_dir = DMSO_SEARCH / "structures"
os.makedirs(output_dir, exist_ok=True)

failed = []

for idx, row in top50.iterrows():
    cas = row["CAS #"]
    smiles = row["solute_smiles"]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES for CAS {cas}, no image generated.")
        failed.append(cas)
        continue

    # File name = CAS ("/" is replaced, it would break the file name)
    safe_cas = str(cas).replace("/", "-")
    image_path = os.path.join(output_dir, f"{safe_cas}.png")

    Draw.MolToFile(mol, image_path, size=(400, 400))

print(f"\nImages generated: {len(top50) - len(failed)}/{len(top50)}")
if failed:
    print(f"Failed for CAS: {failed}")
print(f"Folder: {output_dir}")
