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

# Recharger le fichier source pour récupérer la colonne CAS
df_source = pd.read_csv(EPFL_INVENTORY, encoding='latin-1')
df_source["CAS #"] = df_source["CAS #"].astype(str).str.strip()
df_molecules_source = df_source[
    df_source["CAS #"].notna() & (df_source["CAS #"] != "") & (df_source["CAS #"].str.lower() != "nan")
].copy()

# Charger les SMILES déjà calculés (progression sauvegardée)
smiles_temp = pd.read_csv(EPFL_INVENTORY_SMILES_PROGRESS)

# Recoller CAS et SMILES par position (même ordre de traitement)
df_clean = df_molecules_source.iloc[:len(smiles_temp)].copy()
df_clean["SMILES"] = smiles_temp["SMILES"].values

# Renommer pour matcher ce qu'attend solubility_calculator
df_clean = df_clean.rename(columns={"SMILES": "solute_smiles"})

# Enlever les lignes où le SMILES n'a pas été trouvé
df_clean = df_clean[df_clean["solute_smiles"].notna()].copy()
print(f"Molécules avec SMILES valide : {len(df_clean)}")

# Calcul du logS dans le DMSO
dmso_smiles = "CS(=O)C"
resultats_dmso = solubility_calculator(df_clean, solvent_name="DMSO", solvent_smiles=dmso_smiles)
df_clean = df_clean.join(resultats_dmso)

top50 = df_clean.sort_values("predicted_logS_DMSO", ascending=True).head(50)
top50.to_excel(ensure_dir(DMSO_SEARCH / "top50_dmso_insoluble.xlsx"), index=False)

# --- NOUVEAU : génération des images de structure avec RDKit ---
output_dir = DMSO_SEARCH / "structures"
os.makedirs(output_dir, exist_ok=True)

failed = []

for idx, row in top50.iterrows():
    cas = row["CAS #"]
    smiles = row["solute_smiles"]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"SMILES invalide pour CAS {cas}, image non générée.")
        failed.append(cas)
        continue

    # Nom de fichier = CAS (on remplace "/" au cas où, ça casserait le nom de fichier)
    safe_cas = str(cas).replace("/", "-")
    image_path = os.path.join(output_dir, f"{safe_cas}.png")

    Draw.MolToFile(mol, image_path, size=(400, 400))

print(f"\nImages générées : {len(top50) - len(failed)}/{len(top50)}")
if failed:
    print(f"Échecs pour les CAS : {failed}")
print(f"Dossier : {output_dir}")