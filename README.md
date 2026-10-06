# Project 1B - Solubility Prediction NMR
Predicting deuterated NMR solvent solubility rankings for organic molecules using only HPLC retention data, no structural information (SMILES, fingerprints) required.

Developed as part of Project 1B within the [SwissCat+](https://swisscat.org/) research platform at EPFL.

## Motivation

Choosing a deuterated NMR solvent for a new compound is usually guesswork or trial-and-error. This project asks: **can we predict which NMR solvents a molecule will dissolve in using only its HPLC retention behavior?**

The constraint is no SMILES, no molecular fingerprints, no structural descriptors. Only what an HPLC column and method output (retention time, method metadata) are used as input. This makes it a genuinely blind prediction problem, closer to what's available in a real lab workflow before any structural characterization is done.

## Data

- **151** unique molecules
- **5** deuterated NMR solvents: MeOH, ACN, DMSO, DCM, CHCl₃
- **7** HPLC methods
- **16** input features derived from HPLC retention data and method characterization
- Labels are **real experimental solubility outcomes** (not predicted logS values), which makes the prediction task harder but scientifically honest

## Model

Final model: **XGBoost** ("Model E" internally), selected after comparison against CatBoost (which underperformed despite tuning — eval metric issues on imbalanced classes, eval_set leakage, and early-stopping instability were all encountered and addressed).

### Performance (test set, 22 held-out molecules)

| Metric | Value |
|---|---|
| Accuracy | ~72% |
| Precision | ~82% |
| Recall | ~77% |
| Critical failure rate (per-molecule ranking) | 13.6% |

Splits are done with `GroupShuffleSplit` grouped by molecule (SMILES) to prevent data leakage between train and test sets.

### Feature importance

Top feature by Gini importance: **Molecular Weight (~17.8%)** alone, but all Matrix features combined give a **~30%** importance. SHAP analysis confirms the model recovers real, interpretable chemistry:
- A hydrophobicity–retention time relationship
- A molecular weight "cliff" around ~300 Da

## Usage

```bash
pip install -r requirements.txt
python scripts/predict.py --input examples/example_input.csv
```

The script:
- Accepts a CSV of molecules with HPLC retention data
- Validates required columns against the model's expected feature set (`model.feature_names_in_`)
- Loops over molecules and predicts a solvent solubility ranking for each
- Handles missing solvents gracefully


## Repository structure

Every path is defined once in `src/config.py`. Scripts and notebooks import it, so the
project runs from any machine without editing paths.

```
.
├── data/
│   ├── raw/                 source files, never modified
│   │   ├── hplc_rt_solubility.xlsx        RT + experimental solubility (main dataset)
│   │   ├── swisscat_hplc_database.csv     full SwissCat+ HPLC database
│   │   ├── hplc_matrix_ids.xlsx           HPLC column/method matrix
│   │   └── epfl_inventory.csv             EPFL chemical inventory (not on git)
│   ├── interim/             cleaned / enriched tables (SMILES, unique CAS, fastsolv logS…)
│   ├── processed/           ML-ready data: train.csv, test.csv, matrix_ids.joblib
│   └── archive/             old copies of the inventory (not on git)
│
├── models/
│   ├── xgb_model_e_16features.joblib     final model (Model E), used by scripts/predict.py
│   ├── xgb_hansen_6features.joblib       model trained in notebooks/02_train_model.ipynb
│   └── archive/catboost_logs/            CatBoost training logs (abandoned model)
│
├── notebooks/               01_split_data → 02_train_model → 03_evaluate_model → 04_robustness_check
│
├── src/                     reusable code (library)
│   ├── config.py            all project paths
│   ├── main.py              fastsolv analysis pipeline
│   ├── controller/          feature generation, training helpers
│   ├── data_converter/      HPLC data cleaning, SMILES, descriptors
│   ├── analysis/            plots, ablations, learning curves
│   └── interface/           SMILES input interface (prototype)
│
├── scripts/                 runnable entry points
│   ├── predict.py           rank NMR solvents for new molecules
│   ├── data_prep/           add_new_molecules.py, fetch_smiles_from_cas.py
│   ├── dmso_search/         find_dmso_insoluble.py (candidates from EPFL inventory)
│   └── experiments/         random-forest experiments (fingerprint / RT / logS)
│
├── results/
│   ├── figures/             eda/, fastsolv_errors/, learning_curves/, ablation/, experiments/
│   ├── reports/             text outputs: ablation/, learning_curves/, experiments/
│   └── dmso_search/         top-50 DMSO-insoluble candidates + structure images
│
├── examples/example_input.csv    input format for scripts/predict.py
├── sandbox/                 debug dumps (not on git)
├── requirements.txt
├── README.md
└── LICENSE
```

Run scripts from the project root, e.g. `python src/main.py` or
`python scripts/experiments/rf_fingerprint_rt.py`.

<details>
<summary>Old → new file names</summary>

| Old | New |
|---|---|
| `data/Fichier final (RT+sol).xlsx` | `data/raw/hplc_rt_solubility.xlsx` |
| `data/Fichier final (RT+sol)_MAJ.xlsx` | `data/interim/hplc_rt_solubility_updated.xlsx` |
| `data/Fichier final (RT+sol)_with_smiles.{csv,xlsx}` | `data/interim/hplc_rt_solubility_smiles.{csv,xlsx}` |
| `data/Fichier final (RT+sol)_with_smiles_Unique_CAS[_Unique_CAS].xlsx` | `data/interim/molecules_unique.xlsx` |
| `data/Fichier final (RT+sol)_with_smiles_Unique_CAS_MAJ.xlsx` | `data/interim/molecules_unique_updated.xlsx` |
| `data/Master_Solubility_Matrix[_Unique_CAS].xlsx` | `data/interim/fastsolv_predictions[_unique].xlsx` |
| `data/Compared_Results.xlsx` | `data/interim/fastsolv_vs_experiment.xlsx` |
| `data/VERIFIED_df_desc.xlsx` | `data/interim/features_descriptors.xlsx` |
| `src/dev/Chemistry_Project_Final_Data.xlsx` | `data/interim/features_all_solvents.xlsx` |
| `data/final_database.csv` | `data/raw/swisscat_hplc_database.csv` |
| `data/matrixIDs.xlsx` | `data/raw/hplc_matrix_ids.xlsx` |
| `data/find_DMSO_insol_mol/containers.csv` | `data/raw/epfl_inventory.csv` |
| `data/find_DMSO_insol_mol/containers_with_smiles.csv` | `data/interim/epfl_inventory_smiles.csv` |
| `data/smiles_progress_temp.csv` | `data/interim/epfl_inventory_smiles_progress.csv` |
| `data/ML_Ready_Dataset.csv` | `data/processed/ml_ready_dataset.csv` |
| `data/all_matrix_ids.joblib` | `data/processed/matrix_ids.joblib` |
| `src/dev/final_model.joblib` | `models/xgb_model_e_16features.joblib` |
| `data/xgboost_all_solvents_model.joblib` | `models/xgb_hansen_6features.joblib` |
| `src/dev/script.py` | `scripts/predict.py` |
| `src/dev/find_new_data.py` | `scripts/data_prep/add_new_molecules.py` |
| `src/dev/smilesparsing.py` | `scripts/data_prep/fetch_smiles_from_cas.py` |
| `src/dev/DMSOinsolubleatEPFL.py` | `scripts/dmso_search/find_dmso_insoluble.py` |
| `src/analysis/model_analysis_RT.py` | `scripts/experiments/rf_fingerprint_rt.py` |
| `src/analysis/model_analysis_logS_RT.py` | `scripts/experiments/rf_fingerprint_rt_logs.py` |
| `src/analysis/model_analysis_fingerprint_solvent.py` | `scripts/experiments/rf_fingerprint_solvent.py` |
| `src/split_data.ipynb`, `training.ipynb`, `testing_model.ipynb`, `robustnesscheck.ipynb` | `notebooks/01_…` to `04_…` |
| `src/dev/test_molecules.csv` | `examples/example_input.csv` |

</details>

### Main Weaknesses

- **DMSO class imbalance (297:1).** Only a handful of DMSO-insoluble examples exist (DMSO is 99% soluble across the dataset), so the model has almost nothing to learn the insoluble class from. SMOTE, oversampling, and class weights were all tried and all failed. DMSO chemically dissolves nearly everything, so this is **data-bound, not model-bound**. Targeted measurements of known DMSO-insoluble molecules are the only thing that would unblock this class.
- **~72% feature ceiling.** This is the current absolute accuracy ceiling. The learning curve confirms more data won't push past it: RT and MolWt encode hydrophobicity, not the full solubility picture. Getting past this requires new features, not more rows, nevertheless peak shape (asymmetry and width) is a possible leading candidate, since it likely carries column–molecule interaction information that RT alone misses.
- **13.6% critical failure rate.** 3 of 22 test molecules had the model's top-ranked solvent turn out to be insoluble, an honest failure mode, not hidden. (86.4% of the time the top pick does work, and 68.2% of the time the full ranking is exactly right.)
- **Possible selection-bias in per-solvent accuracy.** When a solvent shows higher held-out accuracy, is that because the model is genuinely better for that solvent, or simply because most of the molecules tested against it happen to be soluble (making the prediction easier by base rate)? Not yet disentangled — worth checking before presenting per-solvent accuracy as a measure of model quality.

### Changes to make

- [ ] Collect targeted experimental data for known DMSO-insoluble molecules to address the 297:1 class imbalance.
- [ ] Engineer peak-shape features (asymmetry, width) as the next lever past the accuracy ceiling.
- [ ] Check whether per-solvent accuracy differences reflect genuine model performance or just base-rate solubility differences across solvents.

Work in progress...

## License

MIT — see [LICENSE](LICENSE).
