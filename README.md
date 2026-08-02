# Solubility Prediction NMR
Predicting deuterated NMR solvent solubility rankings for organic molecules using only HPLC retention data — no structural information (SMILES, fingerprints, etc.) required.

Developed as part of Project 1B within the [SwissCat+](https://swisscat.org/) research platform at EPFL.

## Motivation

Choosing a deuterated NMR solvent for a new compound is usually guesswork or trial-and-error. This project asks: **can we predict which NMR solvents a molecule will dissolve in using only its HPLC retention behavior?**

The constraint is intentional — no SMILES, no molecular fingerprints, no structural descriptors. Only what an HPLC column and method output (retention time, method metadata) are used as input. This makes it a genuinely blind prediction problem, closer to what's available in a real lab workflow before any structural characterization is done.

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

Top feature by Gini importance: **Molecular Weight (~17.8%)**. SHAP analysis confirms the model recovers real, interpretable chemistry:
- A hydrophobicity–retention time relationship
- A molecular weight "cliff" around ~300 Da

### Known limitations

- **DMSO class imbalance**: only 3 DMSO-insoluble examples exist in the dataset (a 297:1 ratio). Neither SMOTE nor class reweighting resolves this — it's a data collection problem, not a model problem.
- **Feature ceiling**: validation learning curves are flat with respect to data volume, suggesting the current features (retention time, molecular weight, etc.) encode hydrophobicity but not the full picture of solubility. More *same-distribution* data is unlikely to help; more *targeted* data (e.g. more DMSO-insoluble examples) likely would.

## Usage

```bash
python script.py --input your_molecules.csv
```

The script:
- Accepts a CSV of molecules with HPLC retention data
- Validates required columns against the model's expected feature set (`model.feature_names_in_`)
- Loops over molecules and predicts a solvent solubility ranking for each
- Handles missing solvents gracefully

## Environment

```bash
conda env create -f environment.yml
conda activate ml_env
```

If using VS Code, make sure `.vscode/settings.json` points at the `ml_env` interpreter — using the wrong environment has previously caused segfaults on load.

## Repository structure

```
.
├── src/
│   └── main.py
├── script.py
├── README.md
└── LICENSE
```

## Open questions

- The precise definition of the 10 matrix vector dimensions used in HPLC column/method characterization is not fully documented in any published source. Pending clarification from Prof. Pascal Miéville (SwissCat+ Operational Director, EPFL).

## License

MIT — see [LICENSE](LICENSE).
