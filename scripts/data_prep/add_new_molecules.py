"""
Builds the updated dataset in 3 steps. Raw files are never modified.

  1. NEW MOLECULES - adds the molecules of the SwissCat database that are not
     yet in the main file (matched by CAS). One row per new molecule:
       - the BlueBird / method_1 row if its retention time (RT) is not 0
       - otherwise the first row with RT != 0, following FALLBACK_ORDER
     Molecules with no usable RT anywhere are skipped and listed.

  2. SOLUBILITY LABELS - fills the solvent columns (1 = soluble, 0 = insoluble)
     from the plate experiments in data/raw/experiments/*.xlsx (sheet "Resultats").
     Label of a well = "Mon observation" if filled, otherwise the vision-model
     "prediction" (the "Exclus" column belongs to another project and is ignored).
     Cells that already contain a label are never overwritten (conflicts are listed).
     -> data/interim/hplc_rt_solubility_updated.xlsx

  3. TRAINING TABLE - one row per CAS with its SMILES: the BlueBird 1 row if it has a
     signal, otherwise the first other HPLC column with a signal (FALLBACK_ORDER).
     SMILES given by hand in data/raw/manual_smiles.csv (columns CAS,SMILES) come
     first, then the SMILES already known in data/interim; the missing ones are
     fetched from PubChem, by CAS then by compound name (internet needed,
     skip with --no-pubchem).
     -> data/interim/molecules_unique_updated.xlsx

  4. MODEL E TABLE - Arthur's format: one row per molecule x HPLC column. Original
     measurements + every HPLC column of the SwissCat database for the plate molecules.
     -> data/interim/model_e_dataset.xlsx (used by notebooks/05_train_model_e.ipynb)

Usage (from the project root):
    python scripts/data_prep/add_new_molecules.py
    python scripts/data_prep/add_new_molecules.py --no-pubchem
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # make src/ importable

import pandas as pd
from openpyxl import load_workbook

# ======================= CONFIGURATION =======================
from config import (SWISSCAT_DATABASE, HPLC_RT_SOLUBILITY, HPLC_RT_SOLUBILITY_UPDATED, EXPERIMENTS,
                    MOLECULES_UNIQUE, MOLECULES_UNIQUE_UPDATED, HPLC_RT_SOLUBILITY_SMILES,
                    MATRIX_IDS_TABLE, MODEL_E_DATASET, ensure_dir)

DATABASE = SWISSCAT_DATABASE
FINAL_FILE = HPLC_RT_SOLUBILITY
OUTPUT = HPLC_RT_SOLUBILITY_UPDATED
TRAINING_TABLE = MOLECULES_UNIQUE_UPDATED
MANUAL_SMILES = Path(HPLC_RT_SOLUBILITY).parent / "manual_smiles.csv"   # optional, columns CAS,SMILES

# Preferred row
PREFERRED = ("BlueBird", "method_1")

# Fallback order when the preferred row has RT = 0 (first match with RT != 0 wins)
FALLBACK_ORDER = [
    ("PFP", "method_1"), ("PolarTec", "method_1"), ("RP18", "method_1"), ("Sphinx", "method_1"),
    ("PFP", "method_2"), ("PolarTec", "method_2"), ("RP18", "method_2"), ("Sphinx", "method_2"),
    ("PFP", "method_3"), ("PolarTec", "method_3"), ("RP18", "method_3"),
]

# Column mapping: database column -> final file column
MAPPING = {
    "column": "Column",
    "method": "Method",
    "Sample Name": "Sample Name",
    "CAS #": "CAS ",            # note: trailing space in the final file
    "RT": "RT",
    "Signal": "Signal",
    "Matrix ID Name": "Matrix ID Name",
}
CAS_COL_DB = "CAS #"
CAS_COL_FINAL = "CAS "
MATRIX_PREFIX = "matrix ID "    # "PFP 1" -> "matrix ID PFP 1"

# Plate experiments
SOLVENTS = ["MeOH", "ACN", "DMSO", "DCM", "CHCl3"]
SOLVENT_ALIASES = {             # name written in the experiment file -> column of the main file
    "meoh": "MeOH", "methanol": "MeOH",
    "acn": "ACN", "mecn": "ACN", "ch3cn": "ACN", "acetonitrile": "ACN",
    "dmso": "DMSO",
    "dcm": "DCM", "ch2cl2": "DCM", "dichloromethane": "DCM",
    "chcl3": "CHCl3", "chloroform": "CHCl3",
}
LABELS = {"soluble": 1, "s": 1, "insoluble": 0, "i": 0}
RT_TOLERANCE = 0.01             # min; larger gaps between experiment RT and file RT are reported
# =============================================================


def norm_cas(x):
    return None if pd.isna(x) else str(x).strip()


def is_filled(x):
    return not (x is None or (isinstance(x, float) and pd.isna(x)) or str(x).strip() == "")


# ------------------------------------------------------------------ step 1
def pick_row(rows):
    """Return (row, used_fallback) for one molecule, or (None, False) if no usable RT."""
    def find(column, method):
        m = rows[(rows["_col"] == column) & (rows["_met"] == method) & (rows["_rt"] != 0)
                 & rows["_rt"].notna()]
        return None if m.empty else m.iloc[0]

    row = find(*PREFERRED)
    if row is not None:
        return row, False
    for column, method in FALLBACK_ORDER:
        row = find(column, method)
        if row is not None:
            return row, True
    return None, False


def add_database_molecules(ws, final):
    db = pd.read_csv(DATABASE, sep=None, engine="python")
    db["_cas"] = db[CAS_COL_DB].map(norm_cas)
    db["_col"] = db["column"].astype(str).str.strip()
    db["_met"] = db["method"].astype(str).str.strip()
    db["_rt"] = pd.to_numeric(db["RT"], errors="coerce")
    db = db.dropna(subset=["_cas"])

    existing = set(final[CAS_COL_FINAL].map(norm_cas).dropna())
    new_cas = [c for c in db["_cas"].unique() if c not in existing]

    selected, fallbacks, no_rt = [], [], []
    for cas in new_cas:
        rows = db[db["_cas"] == cas]
        row, used_fallback = pick_row(rows)
        if row is None:
            no_rt.append((rows["Sample Name"].iloc[0], cas))
            continue
        selected.append(row)
        if used_fallback:
            fallbacks.append((row["Sample Name"], cas, row["_col"], row["_met"], row["_rt"]))

    headers = [c.value for c in ws[1]]
    for row in selected:
        values = {col_final: row[col_db] for col_db, col_final in MAPPING.items()}
        values["Matrix ID Name"] = MATRIX_PREFIX + str(row["Matrix ID Name"]).strip()
        values[CAS_COL_FINAL] = row["_cas"]
        out = []
        for h in headers:
            v = values.get(h)
            out.append(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
        ws.append(out)

    print("===== 1. NEW MOLECULES FROM THE DATABASE =====")
    print(f"Molecules in database      : {db['_cas'].nunique()}")
    print(f"Already in final file      : {db['_cas'].nunique() - len(new_cas)}")
    print(f"New molecules added        : {len(selected)}")
    print(f"  - {PREFERRED[0]} / {PREFERRED[1]}      : {len(selected) - len(fallbacks)}")
    print(f"  - other column (RT = 0 on {PREFERRED[0]}): {len(fallbacks)}")
    for name, cas, col, met, rt in fallbacks:
        print(f"      {name} ({cas}) -> {col} / {met}, RT = {rt}")
    if no_rt:
        print(f"Skipped (RT = 0 everywhere): {len(no_rt)}")
        for name, cas in no_rt:
            print(f"      {name} ({cas})")


# ------------------------------------------------------------------ step 2
def experiment_files():
    """Plate files in data/raw/experiments (Excel lock files '~$...' are skipped)."""
    return sorted(p for p in Path(EXPERIMENTS).glob("*.xlsx") if not p.name.startswith("~$"))


def read_experiment(path):
    """Return one record per usable well: CAS, compound, solvent, label, RT, source."""
    df = pd.read_excel(path, sheet_name="Resultats")
    df.columns = df.columns.astype(str).str.strip()
    # Accept small variations of the column names ("CAS #", "N° CAS", "Composé", "Solvant"...)
    wanted = {"CAS": lambda c: "cas" in c, "compose": lambda c: c.startswith("compos"),
              "solvant": lambda c: c.startswith("solvant") or c.startswith("solvent")}
    for name, match in wanted.items():
        if name not in df.columns:
            found = [c for c in df.columns if match(c.lower())]
            if not found:
                raise KeyError(f"{path.name}: no '{name}' column in the 'Resultats' sheet "
                               f"(columns found: {list(df.columns)})")
            df = df.rename(columns={found[0]: name})
    df = df.dropna(subset=["compose", "solvant"])
    column_col = next((c for c in df.columns if c.lower().startswith(("colonne", "column"))), None)
    warnings = []

    # Excel auto-fill bug: the CAS is incremented down the plate (13055-63-5, -6, -7...).
    # Every well of a compound gets the CAS of its first well.
    df["CAS"] = df["CAS"].map(norm_cas)
    first_cas = df.groupby("compose")["CAS"].transform("first")
    n_fixed = (df["CAS"] != first_cas).sum()
    if n_fixed:
        warnings.append(f"{n_fixed} CAS corrected (Excel auto-increment), first well of each compound kept")
    df["CAS"] = first_cas
    if "RT" in df:
        df["RT"] = pd.to_numeric(df["RT"], errors="coerce").groupby(df["compose"]).transform("first")
    if column_col:
        df[column_col] = df.groupby("compose")[column_col].transform("first")

    observed = df["Mon observation"] if "Mon observation" in df else pd.Series(None, index=df.index)
    label_txt = observed.where(observed.map(is_filled), df["prediction"])

    records = []
    for i, r in df.iterrows():
        well = r.get("puits", i)
        solvent = SOLVENT_ALIASES.get(str(r["solvant"]).strip().lower())
        label = LABELS.get(str(label_txt[i]).strip().lower())
        if solvent is None or label is None:
            warnings.append(f"well {well}: unknown solvent '{r['solvant']}' or label '{label_txt[i]}', skipped")
            continue
        records.append({"cas": r["CAS"], "name": r["compose"], "solvent": solvent, "label": label,
                        "rt": r.get("RT"), "column": r.get(column_col) if column_col else None,
                        "source": f"{path.name}:{well}", "from_observation": is_filled(observed[i])})
    return records, warnings


def plate_matrix(value):
    """'BlueBird' -> ('BlueBird', 'method_1', 'BlueBird 1'); 'PFP 2' -> ('PFP', 'method_2', 'PFP 2')."""
    if not is_filled(value):
        return None
    parts = str(value).strip().split()
    known = {c.lower(): c for c, _ in [PREFERRED] + FALLBACK_ORDER}
    column = known.get(parts[0].lower())
    if column is None:
        return None
    number = parts[1] if len(parts) > 1 and parts[1].isdigit() else "1"
    return column, f"method_{number}", f"{column} {number}"


def add_plate_only_molecules(ws, exp, headers):
    """Molecules of a plate that are neither in the file nor in the SwissCat database:
    add one row with the RT and HPLC column written in the plate file (method 1 when not given)."""
    in_file = {norm_cas(r[headers.index(CAS_COL_FINAL)].value) for r in ws.iter_rows(min_row=2)}
    added = []
    for cas, g in exp.dropna(subset=["cas"]).groupby("cas"):
        if cas in in_file:
            continue
        rt = pd.to_numeric(g["rt"], errors="coerce").dropna()
        matrix = next((m for m in map(plate_matrix, g["column"]) if m), None)
        if rt.empty or rt.iloc[0] <= 0 or matrix is None:
            continue
        column, method, matrix_name = matrix
        values = {"Column": column, "Method": method, "Sample Name": str(g["name"].iloc[0]).strip(),
                  CAS_COL_FINAL: cas, "RT": float(rt.iloc[0]), "Signal": 1, "Matrix ID Name": MATRIX_PREFIX + matrix_name}
        ws.append([values.get(h) for h in headers])
        added.append(f"{values['Sample Name']} ({cas}) -> {matrix_name}, RT = {values['RT']}")
    return added


def fill_from_experiments(ws):
    print("\n===== 2. SOLUBILITY LABELS FROM PLATE EXPERIMENTS =====")
    files = experiment_files()
    if not files:
        print(f"No experiment file in {EXPERIMENTS} - nothing to fill.")
        return

    records = []
    for path in files:
        recs, warnings = read_experiment(path)
        n_obs = sum(r["from_observation"] for r in recs)
        print(f"{path.name}: {len(recs)} wells used ({n_obs} from 'Mon observation', "
              f"{len(recs) - n_obs} from the vision prediction)")
        for w in warnings:
            print(f"    note: {w}")
        records += recs

    # One label per (CAS, solvent); disagreeing wells -> not used
    exp = pd.DataFrame(records)
    labels, disagreements = {}, []
    for (cas, solvent), g in exp.groupby(["cas", "solvent"]):
        if g["label"].nunique() > 1:
            disagreements.append(f"{g['name'].iloc[0]} ({cas}) in {solvent}: "
                                 + ", ".join(f"{s}={l}" for s, l in zip(g["source"], g["label"])))
        else:
            labels[(cas, solvent)] = int(g["label"].iloc[0])

    headers = [c.value for c in ws[1]]
    col = {h: i for i, h in enumerate(headers)}
    added_from_plate = add_plate_only_molecules(ws, exp, headers)
    filled, conflicts, rows_by_cas = 0, [], {}
    for row in ws.iter_rows(min_row=2):
        cas = norm_cas(row[col[CAS_COL_FINAL]].value)
        if cas is None:
            continue
        rows_by_cas.setdefault(cas, []).append(row[col["RT"]].value)
        for solvent in SOLVENTS:
            key = (cas, solvent)
            if key not in labels:
                continue
            cell = row[col[solvent]]
            if not is_filled(cell.value):
                cell.value = labels[key]
                filled += 1
            elif int(cell.value) != labels[key]:
                conflicts.append(f"{cas} in {solvent}: file = {int(cell.value)}, experiment = {labels[key]} (file kept)")

    missing = sorted({c for c, _ in labels} - set(rows_by_cas))
    no_cas = sorted(exp.loc[exp["cas"].isna(), "name"].astype(str).str.strip().unique())
    rt_gaps = []
    for cas, g in exp.groupby("cas"):
        rts = [rt for rt in rows_by_cas.get(cas, []) if rt is not None]
        rt_exp = g["rt"].dropna()
        if rts and len(rt_exp) and min(abs(float(rt_exp.iloc[0]) - rt) for rt in rts) > RT_TOLERANCE:
            rt_gaps.append(f"{g['name'].iloc[0]} ({cas}): experiment RT {rt_exp.iloc[0]}, file RT {rts}")

    print(f"Labels (molecule x solvent): {len(labels)} -> {filled} cells filled")
    if added_from_plate:
        print(f"Not in the HPLC database, RT and column taken from the plate file: {len(added_from_plate)}")
        for it in added_from_plate:
            print(f"    {it}")
    for title, items in [("Wells that disagree (not used)", disagreements),
                         ("Already labelled differently in the file", conflicts),
                         ("Compounds without CAS or RT anywhere (not used)", missing + no_cas),
                         ("RT different from the file", rt_gaps)]:
        if items:
            print(f"{title}: {len(items)}")
            for it in items:
                print(f"    {it}")


# ------------------------------------------------------------------ step 3
def known_smiles():
    """CAS -> SMILES: manual file first, then the tables that already have SMILES."""
    known = {}
    if MANUAL_SMILES.exists():
        manual = pd.read_csv(MANUAL_SMILES)
        manual.columns = manual.columns.str.strip()
        for cas, smi in zip(manual["CAS"].map(norm_cas), manual["SMILES"]):
            if cas and is_filled(smi):
                known[cas] = str(smi).strip()
        print(f"SMILES given by hand       : {len(known)} ({MANUAL_SMILES.name})")
    for path in [MOLECULES_UNIQUE, MOLECULES_UNIQUE_UPDATED,
                 HPLC_RT_SOLUBILITY_SMILES, HPLC_RT_SOLUBILITY_SMILES.with_suffix(".csv")]:
        if not Path(path).exists():
            continue
        df = pd.read_excel(path) if str(path).endswith(".xlsx") else pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if "solute_smiles" not in df:
            continue
        for cas, smi in zip(df["CAS"].map(norm_cas), df["solute_smiles"]):
            if cas and is_filled(smi) and smi != "Did not work":
                known.setdefault(cas, smi)
    return known


def build_training_table(use_pubchem):
    print("\n===== 3. TRAINING TABLE (one row per CAS, with SMILES) =====")
    df = pd.read_excel(OUTPUT)
    df = df.dropna(subset=[CAS_COL_FINAL])
    df[CAS_COL_FINAL] = df[CAS_COL_FINAL].map(norm_cas)
    # one row per molecule: BlueBird 1 if it has a signal, otherwise the next column in FALLBACK_ORDER
    df = df[(df["Signal"] != 0) & (pd.to_numeric(df["RT"], errors="coerce") > 0)].copy()
    priority = {PREFERRED: 0, **{cm: i + 1 for i, cm in enumerate(FALLBACK_ORDER)}}
    df["_prio"] = [priority.get((str(c).strip(), str(m).strip()), 99) for c, m in zip(df["Column"], df["Method"])]
    df = df.sort_values("_prio", kind="stable").drop_duplicates(subset=[CAS_COL_FINAL], keep="first")
    other = df[df["_prio"] > 0]
    print(f"Molecules without BlueBird 1 signal, other column used: {len(other)}")
    for _, r in other.iterrows():
        print(f"    {r['Sample Name']} ({r[CAS_COL_FINAL]}) -> {r['Matrix ID Name']}, RT = {r['RT']}")
    df = df.drop(columns="_prio").sort_index()

    known = known_smiles()
    df["solute_smiles"] = df[CAS_COL_FINAL].map(known)
    todo = df["solute_smiles"].isna() & df[SOLVENTS].notna().any(axis=1)  # only labelled molecules
    print(f"SMILES already known       : {df['solute_smiles'].notna().sum()}")
    if todo.any():
        if use_pubchem:
            from data_converter.hplc_data_handler import smiles_by_pubchem_cas
            print(f"Fetching {todo.sum()} SMILES from PubChem...")
            def fetch(cas, name):
                # by CAS, then by compound name, then CAS again (PubChem sometimes times out)
                for query in (cas, name, cas):
                    smi = smiles_by_pubchem_cas(query) if is_filled(query) else None
                    if is_filled(smi) and smi != "Did not work":
                        return smi
                return None
            df.loc[todo, "solute_smiles"] = [fetch(c, n) for c, n in zip(df.loc[todo, CAS_COL_FINAL], df.loc[todo, "Sample Name"])]
        else:
            print(f"--no-pubchem: {todo.sum()} labelled molecules left without SMILES (ignored by training)")

    df.to_excel(ensure_dir(TRAINING_TABLE), index=False)
    usable = df["solute_smiles"].notna() & df[SOLVENTS].notna().any(axis=1)
    print(f"Molecules in table         : {len(df)}")
    print(f"Usable for training        : {usable.sum()} (SMILES + at least one solubility label)")
    no_smiles = df[df["solute_smiles"].isna() & df[SOLVENTS].notna().any(axis=1)]
    for _, r in no_smiles.iterrows():
        print(f"    no SMILES: {r['Sample Name']} ({r[CAS_COL_FINAL]})")
    if len(no_smiles):
        print(f"    -> add them by hand in {MANUAL_SMILES} (columns CAS,SMILES) and run the script again")
    print(f"Saved to: {TRAINING_TABLE}")


# ------------------------------------------------------------------ step 4
def build_model_e_table():
    """Arthur's Model E input (one row per molecule x HPLC column, labels per solvent, SMILES):
    the original measurements + every HPLC column of the SwissCat database for the plate molecules."""
    print("\n===== 4. MODEL E TABLE (all HPLC columns, Arthur's format) =====")
    smiles = known_smiles()                         # all 151 original molecules + manual SMILES
    table_smiles = pd.read_excel(TRAINING_TABLE)
    table_smiles.columns = table_smiles.columns.str.strip()
    for cas, smi in zip(table_smiles["CAS"].map(norm_cas), table_smiles["solute_smiles"]):
        if cas and is_filled(smi):
            smiles.setdefault(cas, smi)             # + the ones fetched from PubChem in step 3

    updated = pd.read_excel(OUTPUT)
    updated[CAS_COL_FINAL] = updated[CAS_COL_FINAL].map(norm_cas)
    original_cas = set(pd.read_excel(FINAL_FILE)[CAS_COL_FINAL].map(norm_cas).dropna())
    orig_rows = updated[updated[CAS_COL_FINAL].isin(original_cas)].copy()

    # labelled molecules that are not in the original file (plate experiments)
    first = updated.drop_duplicates(CAS_COL_FINAL).set_index(CAS_COL_FINAL)
    new_cas = [c for c in first.index if c not in original_cas and first.loc[c, SOLVENTS].notna().any()]
    known_matrices = set(pd.read_excel(MATRIX_IDS_TABLE)["MatrixID"].astype(str).str.strip())
    db = pd.read_csv(DATABASE, sep=None, engine="python")
    db["_cas"] = db[CAS_COL_DB].map(norm_cas)
    db["RT"] = pd.to_numeric(db["RT"], errors="coerce")
    db["Matrix ID Name"] = db["Matrix ID Name"].astype(str).str.strip()
    db = db[db["_cas"].isin(new_cas) & (db["RT"] > 0) & db["Matrix ID Name"].isin(known_matrices)]
    new_rows = pd.DataFrame({
        "Column": db["column"], "Method": db["method"], "Sample Name": db["Sample Name"], CAS_COL_FINAL: db["_cas"],
        "RT": db["RT"], "Signal": db["Signal"], "Matrix ID Name": MATRIX_PREFIX + db["Matrix ID Name"],
    })
    # plate molecules absent from the database: their row of the updated file (RT and column from the plate)
    plate_only = updated[updated[CAS_COL_FINAL].isin(set(new_cas) - set(db["_cas"]))
                         & updated["Matrix ID Name"].astype(str).str.replace(MATRIX_PREFIX, "", regex=False).str.strip().isin(known_matrices)]
    new_rows = pd.concat([new_rows, plate_only[new_rows.columns]], ignore_index=True)
    for s in SOLVENTS:
        new_rows[s] = new_rows[CAS_COL_FINAL].map(first[s])

    table = pd.concat([orig_rows, new_rows], ignore_index=True)
    table["solute_smiles"] = table[CAS_COL_FINAL].map(smiles)
    table.to_excel(ensure_dir(MODEL_E_DATASET), index=False)
    print(f"Original measurements      : {len(orig_rows)} rows ({orig_rows[CAS_COL_FINAL].nunique()} molecules)")
    print(f"Plate molecules added      : {len(new_rows)} rows ({new_rows[CAS_COL_FINAL].nunique()} molecules, "
          f"{len(new_cas) - new_rows[CAS_COL_FINAL].nunique()} without a usable HPLC measurement)")
    print(f"Saved to: {MODEL_E_DATASET}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-pubchem", action="store_true", help="do not query PubChem for missing SMILES")
    args = parser.parse_args()

    final = pd.read_excel(FINAL_FILE)
    wb = load_workbook(FINAL_FILE)          # openpyxl keeps the original formatting
    ws = wb.active

    add_database_molecules(ws, final)
    fill_from_experiments(ws)
    wb.save(ensure_dir(OUTPUT))
    print(f"Saved to: {OUTPUT}")

    build_training_table(use_pubchem=not args.no_pubchem)
    build_model_e_table()


if __name__ == "__main__":
    main()
