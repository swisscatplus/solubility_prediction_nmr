"""
fastsolv logS for every molecule of the Model E dataset (194 molecules, plates included), in the 5 NMR solvents.

Run from the project root:
    python scripts/data_prep/fastsolv_all_molecules.py
    python scripts/data_prep/fastsolv_all_molecules.py --batch 5     # smaller batches if it runs out of memory
    python scripts/data_prep/fastsolv_all_molecules.py --n-cpus 4    # faster, if the machine has enough memory

Molecules are sent to fastsolv in small batches and the progress is saved after each batch,
so if the run stops, running it again continues where it stopped.

Writes data/interim/fastsolv_predictions_all.xlsx: one row per molecule (CAS, SMILES),
predicted_logS_<solvent> and predicted_logS_stdev_<solvent> for each solvent.
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # make src/ importable

import pandas as pd
from config import MODEL_E_DATASET, INTERIM, FASTSOLV_PREDICTIONS_ALL, ensure_dir
from data_converter.hplc_data_handler import NMR_SOLVENTS

OUTPUT = FASTSOLV_PREDICTIONS_ALL                              # read by notebook 03b (Model E*)
PROGRESS = INTERIM / "fastsolv_predictions_all_progress.csv"   # one row per (molecule, solvent), deleted at the end
TEMPERATURE = 298.15


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=10, help="molecules per fastsolv call (default 10)")
    parser.add_argument("--n-cpus", type=int, default=1, help="processes for the descriptor calculation (default 1)")
    args = parser.parse_args()

    df = pd.read_excel(MODEL_E_DATASET)
    df["CAS"] = df["CAS "].astype(str).str.strip()
    df["SMILES"] = df["solute_smiles"].astype(str).str.strip()
    molecules = df.drop_duplicates("CAS")[["CAS", "Sample Name", "SMILES"]].reset_index(drop=True)

    done = pd.read_csv(PROGRESS, dtype={"CAS": str}) if PROGRESS.exists() else pd.DataFrame()
    todo = molecules[~molecules["CAS"].isin(set(done["CAS"]) if len(done) else set())]
    print(f"{len(molecules)} molecules, {len(molecules) - len(todo)} already done, {len(todo)} to predict")

    from fastsolv import fastsolv   # loads the 4 fastsolv models once
    import fastprop.descriptors     # fastprop starts one process per CPU, each reloading PyTorch:
    fastprop.descriptors._N_CPUS = args.n_cpus   # too much memory on a small WSL, so limit it
    for start in range(0, len(todo), args.batch):
        chunk = todo.iloc[start:start + args.batch]
        pairs = pd.DataFrame([
            {"CAS": cas, "solvent": name, "solute_smiles": smi, "solvent_smiles": solvent_smi, "temperature": TEMPERATURE}
            for cas, smi in zip(chunk["CAS"], chunk["SMILES"])
            for name, solvent_smi in NMR_SOLVENTS.items()
        ])
        result = fastsolv(pairs[["solute_smiles", "solvent_smiles", "temperature"]])
        pairs["predicted_logS"] = result["predicted_logS"].values
        pairs["predicted_logS_stdev"] = result["predicted_logS_stdev"].values

        cols = ["CAS", "solvent", "predicted_logS", "predicted_logS_stdev"]
        pairs[cols].to_csv(ensure_dir(PROGRESS), mode="a", header=not PROGRESS.exists(), index=False)
        print(f"  {min(start + args.batch, len(todo))}/{len(todo)} molecules done")

    # Back to one row per molecule
    all_pairs = pd.read_csv(PROGRESS, dtype={"CAS": str}).drop_duplicates(["CAS", "solvent"])
    wide = all_pairs.pivot(index="CAS", columns="solvent", values=["predicted_logS", "predicted_logS_stdev"])
    wide.columns = [f"{value}_{solvent}" for value, solvent in wide.columns]
    out = molecules.merge(wide.reset_index(), on="CAS", how="left")

    out.to_excel(ensure_dir(OUTPUT), index=False)
    PROGRESS.unlink()
    print(f"Saved {len(out)} molecules -> {OUTPUT}")


if __name__ == "__main__":
    main()
