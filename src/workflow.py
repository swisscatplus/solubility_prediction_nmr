"""
NMR solvent selection workflow: returns the ONE deuterated solvent to use for a purified molecule.

    Case A (structure unknown): Model E (RT + molar mass + HPLC column + solvent properties)
    Case B (SMILES given):      fastsolv logS (by default, see MODEL_WEIGHTS)

Then, for each candidate solvent:
    score = P(soluble) / (PENALTY_OFFSET + sum(weight * penalty))
and the solvent with the highest score is returned.

A failed solvent removes its whole cluster (e.g. CHCl3 failed -> DCM is not proposed either).
If the structure is known and contains an acid-sensitive group (SMARTS), CHCl3 is excluded (DCl traces).

Run from the project root:
    python src/workflow.py --rt 5.6 --mz 186.0 --matrix-id "BlueBird 1" --solvents MeOH ACN DMSO DCM CHCl3
    python src/workflow.py --rt 5.6 --mz 186.0 --matrix-id "BlueBird 1" --solvents MeOH ACN DMSO DCM CHCl3 \
                           --smiles "CC(C1=CC=CC=C1)Br" --failed DMSO

Only the solvent name is printed (or NONE if no solvent is left). Add --verbose to see the details.
It can also be imported:  from workflow import select_solvent
"""
import sys
import math
import argparse
import contextlib

import numpy as np
import pandas as pd
import joblib

from config import MODEL_E, MATRIX_IDS_TABLE


# ======================================================================
# SETTINGS: everything that may need to change is here
# ======================================================================

# --- Models. Case A = no SMILES, case B = SMILES given.
# P(soluble) = weighted mean of the models with a weight > 0.
MODEL_WEIGHTS = {
    "no_smiles": {"model_e": 1.0},
    "smiles":    {"model_e": 0.0, "fastsolv": 1.0},
}

FASTSOLV_THRESHOLD = 0.05      # mol/L: soluble above this (same threshold as src/main.py)
FASTSOLV_TEMPERATURE = 298.15  # K
FASTSOLV_N_CPUS = 1            # processes for fastsolv's descriptors (more = faster but much more memory)
MZ_TO_MASS = -1.007            # [M+H]+  ->  molar mass = m/z - 1.007 (only used without SMILES)

# --- Penalties: score = P / (PENALTY_OFFSET + sum(weight * penalty))
PENALTY_OFFSET = 0.0
PENALTY_WEIGHTS = {
    "cost": 1.0,               # CHF per NMR tube = price * density * TUBE_VOLUME_ML
    "nmr_compatibility": 1.0,  # e.g. DMSO freezes at 18.5 degC
    "solvent_signal": 1.0,     # residual solvent peak in the way
    "water_signal": 1.0,       # hygroscopic solvent -> large water peak
}
TUBE_VOLUME_ML = 0.6

# --- Solvents. The name used here is the one the model knows.
# cluster: solvents that behave alike; when one fails, the whole cluster is skipped.
SOLVENTS = {
    "CHCl3": dict(price=1.0,  density=1.50, cluster="chlorinated",
                  nmr_compatibility=0, solvent_signal=0, water_signal=0,
                  dielectric=4.81, hansen_d=17.8, hansen_p=3.1, hansen_h=5.7, smiles="ClC(Cl)Cl"),
    "DCM":   dict(price=36.0, density=1.36, cluster="chlorinated",
                  nmr_compatibility=0, solvent_signal=0, water_signal=0,
                  dielectric=8.93, hansen_d=18.2, hansen_p=6.3, hansen_h=6.1, smiles="C(Cl)Cl"),
    "DMSO":  dict(price=3.5,  density=1.19, cluster="DMSO",
                  nmr_compatibility=1, solvent_signal=1, water_signal=1,
                  dielectric=46.7, hansen_d=18.4, hansen_p=16.4, hansen_h=10.2, smiles="CS(=O)C"),
    "MeOH":  dict(price=18.0, density=0.89, cluster="polar",
                  nmr_compatibility=0, solvent_signal=0, water_signal=0,
                  dielectric=32.7, hansen_d=15.1, hansen_p=12.3, hansen_h=22.3, smiles="CO"),
    "ACN":   dict(price=15.0, density=0.84, cluster="polar",
                  nmr_compatibility=0, solvent_signal=0, water_signal=0,
                  dielectric=37.5, hansen_d=15.3, hansen_p=18.0, hansen_h=6.1, smiles="CC#N"),
}

# Other names accepted in --solvents / --failed (case-insensitive). The answer uses the caller's spelling.
SOLVENT_ALIASES = {
    "cdcl3": "CHCl3", "chloroform": "CHCl3", "chloroform-d": "CHCl3",
    "cd2cl2": "DCM", "dcm-d2": "DCM", "dichloromethane": "DCM",
    "dmso-d6": "DMSO", "dmsod6": "DMSO",
    "meod": "MeOH", "cd3od": "MeOH", "meoh-d4": "MeOH", "methanol": "MeOH", "methanol-d4": "MeOH",
    "cd3cn": "ACN", "acn-d3": "ACN", "acetonitrile": "ACN", "acetonitrile-d3": "ACN",
}

# --- Acid-sensitive groups (case B only). If one matches, these solvents are excluded.
ACID_EXCLUDED_SOLVENTS = ["CHCl3"]
ACID_SENSITIVE_SMARTS = {
    # from RDKit (rdkit/Data/FragmentDescriptors.csv)
    "imine (fr_Imine)":        "[Nv3](=C)-[#6]",
    "hydrazone (fr_hdrzone)":  "C=N-[NX3]",
    "epoxide (fr_epoxide)":    "O1CC1",
    "diazonium (fr_diazo)":    "[N+]#N",
    # added (not in RDKit's list)
    "tBu ester / Boc":         "[CX4](C)(C)(C)[OX2][CX3]=O",
    "acetal / ketal":          "[CX4]([OX2][#6])[OX2][#6]",
    "hemiacetal":              "[CX4]([OX2H])[OX2][#6]",
    "silyl ether":             "[Si][OX2][#6]",
    "enol ether":              "[#6;!$(C=O)][OX2][CX3;!$(C[CX3]=[OX1])]=[CX3;!$(C[CX3]=[OX1])]",
    "enamine":                 "[NX3;!$(N[C,S]=O)][CX3]=[CX3;!$(C[C,S]=O)]",
    "trityl":                  "[CX4](c)(c)(c)[O,N,S]",
    "aziridine":               "N1CC1",
    "diazo":                   "[CX3]=[N+]=[N-]",
}

# ======================================================================


class WorkflowError(Exception):
    """Wrong input (unknown HPLC column, invalid SMILES...)."""


def warn(message):
    print(f"WARNING: {message}", file=sys.stderr)


# ------------------------------------------------------------- inputs
def canonical_solvent(name):
    """'CDCl3' -> 'CHCl3', 'DMSO' -> 'DMSO', unknown -> None."""
    clean = str(name).strip()
    for key in SOLVENTS:
        if clean.lower() == key.lower():
            return key
    return SOLVENT_ALIASES.get(clean.lower())


def matrix_vector(matrix_id):
    """RTs of the 10 standards on this HPLC column (same table and logic as notebook 01)."""
    table = pd.read_excel(MATRIX_IDS_TABLE)
    name = str(matrix_id).replace("matrix ID ", "").strip()
    m_cols = [c for c in table.columns if c.startswith("M") and c != "MatrixID"]
    row = table[table["MatrixID"].astype(str).str.strip().str.lower() == name.lower()]
    if row.empty:
        raise WorkflowError(f"Unknown MatrixID '{matrix_id}'. Known: {table['MatrixID'].tolist()}")
    return row[m_cols].iloc[0].values.astype(float)


def parse_smiles(smiles):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")        # our own error message is enough
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise WorkflowError(f"Invalid SMILES: '{smiles}'")
    return mol


def acid_sensitive_groups(mol):
    """Names of the acid-sensitive groups found in the molecule."""
    from rdkit import Chem
    found = []
    for name, smarts in ACID_SENSITIVE_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise WorkflowError(f"Invalid SMARTS for '{name}': {smarts}")
        if mol.HasSubstructMatch(pattern):
            found.append(name)
    return found


# ------------------------------------------------------------- models
def proba_model_e(solvents, rt, mol_wt, matrix_id):
    """Model E: P(soluble) for each solvent, exactly as scripts/predict.py."""
    model = joblib.load(MODEL_E)
    vector = matrix_vector(matrix_id)
    rows = []
    for solvent in solvents:
        s = SOLVENTS[solvent]
        row = {"RT": rt, "MolWt": mol_wt,
               "Sol_Dielectric": s["dielectric"], "Sol_Hansen_D": s["hansen_d"],
               "Sol_Hansen_P": s["hansen_p"], "Sol_Hansen_H": s["hansen_h"]}
        row.update({f"Matrix_Vector_{i + 1}": v for i, v in enumerate(vector)})
        rows.append(row)
    X = pd.DataFrame(rows)[list(model.feature_names_in_)]
    return dict(zip(solvents, model.predict_proba(X)[:, 1]))


def proba_fastsolv(solvents, smiles):
    """fastsolv: logS -> P(logS > log10(threshold)), using fastsolv's own uncertainty."""
    # fastsolv / PyTorch print progress bars: send them to stderr so that stdout only holds the answer
    with contextlib.redirect_stdout(sys.stderr):
        try:
            from fastsolv import fastsolv   # imported here: case A works without fastsolv / PyTorch
        except Exception as e:
            raise WorkflowError(f"fastsolv is not available ({e})") from e
        import fastprop.descriptors     # fastprop starts one process per CPU, each reloading PyTorch:
        fastprop.descriptors._N_CPUS = FASTSOLV_N_CPUS   # too much memory on a small machine
        data = pd.DataFrame({
            "solute_smiles": [smiles] * len(solvents),
            "solvent_smiles": [SOLVENTS[s]["smiles"] for s in solvents],
            "temperature": [FASTSOLV_TEMPERATURE] * len(solvents),
        })
        try:
            result = fastsolv(data)
        except Exception as e:
            raise WorkflowError(f"fastsolv failed ({type(e).__name__}: {e})") from e
    log_s = result["predicted_logS"].values.astype(float)
    stdev = np.maximum(result["predicted_logS_stdev"].values.astype(float), 1e-6)
    z = (log_s - math.log10(FASTSOLV_THRESHOLD)) / stdev
    proba = [0.5 * (1 + math.erf(v / math.sqrt(2))) for v in z]   # normal CDF
    return dict(zip(solvents, proba)), dict(zip(solvents, log_s))


# ------------------------------------------------------------- scoring
def penalty_sum(solvent):
    s = SOLVENTS[solvent]
    cost = s["price"] * s["density"] * TUBE_VOLUME_ML
    total = PENALTY_OFFSET + PENALTY_WEIGHTS["cost"] * cost
    for name in ("nmr_compatibility", "solvent_signal", "water_signal"):
        total += PENALTY_WEIGHTS[name] * s[name]
    return total, cost


def select_solvent(rt, mz, matrix_id, solvents, smiles=None, failed=None, verbose=False):
    """Return the name of the solvent to use (spelled as in `solvents`), or None if none is left."""
    failed = failed or []

    # 1. Candidate solvents
    spelling = {}                         # model name -> caller's spelling
    for name in solvents:
        key = canonical_solvent(name)
        if key is None:
            warn(f"solvent '{name}' is unknown to the workflow, ignored")
        else:
            spelling.setdefault(key, str(name).strip())

    removed = {}                          # solvent -> reason
    for name in failed:
        key = canonical_solvent(name)
        if key is None:
            warn(f"failed solvent '{name}' is unknown, ignored")
            continue
        for other, s in SOLVENTS.items():
            if s["cluster"] == SOLVENTS[key]["cluster"]:
                removed.setdefault(other, f"same cluster as failed {name}" if other != key else "failed")

    # 2. Structure known? -> molar mass from the SMILES + acid-sensitive groups
    mol = parse_smiles(smiles) if smiles else None
    if mol is not None:
        from rdkit.Chem import Descriptors
        mol_wt = Descriptors.MolWt(mol)   # as in the notebooks
        groups = acid_sensitive_groups(mol)
        if groups:
            for solvent in ACID_EXCLUDED_SOLVENTS:
                removed.setdefault(solvent, "acid-sensitive: " + ", ".join(groups))
    else:
        mol_wt = float(mz) + MZ_TO_MASS
        groups = []

    candidates = [s for s in spelling if s not in removed]
    if verbose:
        print(f"Case {'B (SMILES)' if mol is not None else 'A (no SMILES)'} | MolWt {mol_wt:.2f} | RT {rt}"
              f" | {matrix_id}", file=sys.stderr)
        for solvent, reason in removed.items():
            if solvent in spelling:
                print(f"  excluded {spelling[solvent]}: {reason}", file=sys.stderr)
    if not candidates:
        return None

    # 3. P(soluble) = weighted mean of the models
    weights = {m: w for m, w in MODEL_WEIGHTS["smiles" if mol is not None else "no_smiles"].items() if w > 0}
    if not weights:
        raise WorkflowError("All model weights are 0 in MODEL_WEIGHTS")
    probas, extra = {}, {}
    if "model_e" in weights:
        probas["model_e"] = proba_model_e(candidates, float(rt), mol_wt, matrix_id)
    if "fastsolv" in weights:
        probas["fastsolv"], extra["logS"] = proba_fastsolv(candidates, smiles)
    p = {s: sum(weights[m] * probas[m][s] for m in weights) / sum(weights.values()) for s in candidates}

    # 4. Penalties and score
    table = []
    for solvent in candidates:
        total, cost = penalty_sum(solvent)
        if total <= 0:
            raise WorkflowError(f"Penalty sum is {total} for {solvent}: check PENALTY_WEIGHTS / PENALTY_OFFSET")
        table.append({"Solvent": spelling[solvent], "P": p[solvent], "Cost_CHF": cost,
                      "Penalty_sum": total, "Score": p[solvent] / total,
                      **{f"P_{m}": probas[m][solvent] for m in probas},
                      **({"logS": extra["logS"][solvent]} if "logS" in extra else {})})
    table = pd.DataFrame(table).sort_values("Score", ascending=False)
    if verbose:
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"), file=sys.stderr)
    return table.iloc[0]["Solvent"]


# ------------------------------------------------------------- command line
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rt", type=float, required=True, help="retention time (min)")
    parser.add_argument("--mz", type=float, required=True, help="measured m/z ([M+H]+)")
    parser.add_argument("--matrix-id", required=True, help='HPLC column + method, e.g. "BlueBird 1"')
    parser.add_argument("--solvents", nargs="+", required=True, help="available NMR solvents")
    parser.add_argument("--smiles", help="structure, if known (case B)")
    parser.add_argument("--failed", nargs="*", default=[], help="solvents already tried without success")
    parser.add_argument("--verbose", "-v", action="store_true", help="print the details (to stderr)")
    args = parser.parse_args()

    try:
        best = select_solvent(args.rt, args.mz, args.matrix_id, args.solvents,
                              smiles=args.smiles, failed=args.failed, verbose=args.verbose)
    except WorkflowError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    print(best if best is not None else "NONE")
    sys.exit(0 if best is not None else 1)


if __name__ == "__main__":
    main()
