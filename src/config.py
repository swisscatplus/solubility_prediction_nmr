"""
Central place for every path in the project.

All scripts, modules and notebooks import their paths from here, so the
project runs on any machine without editing hard-coded paths.

    from config import RAW, INTERIM, PROCESSED, MODELS, FIGURES, REPORTS
"""
from pathlib import Path

# <project>/src/config.py  ->  <project>
ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------- data
DATA = ROOT / "data"
RAW = DATA / "raw"              # source files, never modified
INTERIM = DATA / "interim"      # cleaned / enriched intermediate tables
PROCESSED = DATA / "processed"  # ML-ready datasets and train/test split
ARCHIVE = DATA / "archive"      # old copies kept for reference

# raw
HPLC_RT_SOLUBILITY = RAW / "hplc_rt_solubility.xlsx"          # RT + experimental solubility (main dataset)
SWISSCAT_DATABASE = RAW / "swisscat_hplc_database.csv"        # full SwissCat+ HPLC database
MATRIX_IDS_TABLE = RAW / "hplc_matrix_ids.xlsx"               # HPLC column/method matrix description
EPFL_INVENTORY = RAW / "epfl_inventory.csv"                   # EPFL chemical inventory export

# interim
HPLC_RT_SOLUBILITY_UPDATED = INTERIM / "hplc_rt_solubility_updated.xlsx"
HPLC_RT_SOLUBILITY_SMILES = INTERIM / "hplc_rt_solubility_smiles.xlsx"
MOLECULES_UNIQUE = INTERIM / "molecules_unique.xlsx"          # one row per CAS, with SMILES
MOLECULES_UNIQUE_UPDATED = INTERIM / "molecules_unique_updated.xlsx"
FASTSOLV_PREDICTIONS = INTERIM / "fastsolv_predictions.xlsx"  # fastsolv logS for every solvent
FASTSOLV_VS_EXPERIMENT = INTERIM / "fastsolv_vs_experiment.xlsx"
FEATURES_DESCRIPTORS = INTERIM / "features_descriptors.xlsx"
FEATURES_ALL_SOLVENTS = INTERIM / "features_all_solvents.xlsx"
EPFL_INVENTORY_SMILES = INTERIM / "epfl_inventory_smiles.csv"
EPFL_INVENTORY_SMILES_PROGRESS = INTERIM / "epfl_inventory_smiles_progress.csv"

# processed
ML_READY_DATASET = PROCESSED / "ml_ready_dataset.csv"
TRAIN = PROCESSED / "train.csv"
TEST = PROCESSED / "test.csv"
MATRIX_IDS = PROCESSED / "matrix_ids.joblib"

# --------------------------------------------------------------- models
MODELS = ROOT / "models"
MODEL_E = MODELS / "xgb_model_e_16features.joblib"            # final model described in the README
MODEL_HANSEN = MODELS / "xgb_hansen_6features.joblib"         # model trained in notebooks/02_train_model

# -------------------------------------------------------------- results
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
REPORTS = RESULTS / "reports"
DMSO_SEARCH = RESULTS / "dmso_search"

FIG_EDA = FIGURES / "eda"
FIG_FASTSOLV_ERRORS = FIGURES / "fastsolv_errors"
FIG_LEARNING_CURVES = FIGURES / "learning_curves"
FIG_ABLATION = FIGURES / "ablation"
FIG_EXPERIMENTS = FIGURES / "experiments"

REP_ABLATION_RF = REPORTS / "ablation" / "random_forest"
REP_ABLATION_XGB = REPORTS / "ablation" / "xgboost"
REP_LEARNING_CURVES = REPORTS / "learning_curves"
REP_EXPERIMENTS = REPORTS / "experiments"

EXAMPLES = ROOT / "examples"


def ensure_dir(path):
    """Create the parent folder of a file (or the folder itself) and return the path."""
    path = Path(path)
    (path if path.suffix == "" else path.parent).mkdir(parents=True, exist_ok=True)
    return path
