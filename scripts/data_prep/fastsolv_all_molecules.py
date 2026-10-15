"""
fastsolv logS for every molecule of the Model E dataset (194 molecules, plates included), in the 5 NMR solvents.

Run from the project root:
    python scripts/data_prep/fastsolv_all_molecules.py

Writes data/interim/fastsolv_predictions_all.xlsx: one row per molecule (CAS, SMILES),
predicted_logS_<solvent> and predicted_logS_stdev_<solvent> for each solvent.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # make src/ importable

import pandas as pd
from config import MODEL_E_DATASET, INTERIM, ensure_dir
from data_converter.hplc_data_handler import NMR_SOLVENTS

OUTPUT = INTERIM / "fastsolv_predictions_all.xlsx"
TEMPERATURE = 298.15


def main():
    df = pd.read_excel(MODEL_E_DATASET)
    df["CAS"] = df["CAS "].astype(str).str.strip()
    df["SMILES"] = df["solute_smiles"].astype(str).str.strip()
    molecules = df.drop_duplicates("CAS")[["CAS", "Sample Name", "SMILES"]].reset_index(drop=True)
    print(f"{len(molecules)} molecules, {len(NMR_SOLVENTS)} solvents")

    # One fastsolv call with every (molecule, solvent) pair
    pairs = pd.DataFrame([
        {"CAS": cas, "solvent": name, "solute_smiles": smi, "solvent_smiles": solvent_smi, "temperature": TEMPERATURE}
        for cas, smi in zip(molecules["CAS"], molecules["SMILES"])
        for name, solvent_smi in NMR_SOLVENTS.items()
    ])
    from fastsolv import fastsolv
    result = fastsolv(pairs[["solute_smiles", "solvent_smiles", "temperature"]])
    pairs["predicted_logS"] = result["predicted_logS"].values
    pairs["predicted_logS_stdev"] = result["predicted_logS_stdev"].values

    # Back to one row per molecule
    wide = pairs.pivot(index="CAS", columns="solvent", values=["predicted_logS", "predicted_logS_stdev"])
    wide.columns = [f"{value}_{solvent}" for value, solvent in wide.columns]
    out = molecules.merge(wide.reset_index(), on="CAS", how="left")

    out.to_excel(ensure_dir(OUTPUT), index=False)
    print(f"Saved {len(out)} molecules -> {OUTPUT}")


if __name__ == "__main__":
    main()
