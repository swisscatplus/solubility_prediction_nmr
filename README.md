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
python script.py --input your_molecules.csv
```

The script:
- Accepts a CSV of molecules with HPLC retention data
- Validates required columns against the model's expected feature set (`model.feature_names_in_`)
- Loops over molecules and predicts a solvent solubility ranking for each
- Handles missing solvents gracefully


## Repository structure

```
.
├── src/
│   └── main.py
├── script.py
├── README.md
└── LICENSE
```

### Main Weaknesses

- **DMSO class imbalance (297:1).** Only a handful of DMSO-insoluble examples exist (DMSO is 99% soluble across the dataset), so the model has almost nothing to learn the insoluble class from. SMOTE, oversampling, and class weights were all tried and all failed. DMSO chemically dissolves nearly everything, so this is **data-bound, not model-bound**. Targeted measurements of known DMSO-insoluble molecules are the only thing that would unblock this class.
- **~72% feature ceiling.** This is the current absolute accuracy ceiling. The learning curve confirms more data won't push past it: RT and MolWt encode hydrophobicity, not the full solubility picture. Getting past this requires new features, not more rows, nevertheless peak shape (asymmetry and width) is a possible leading candidate, since it likely carries column–molecule interaction information that RT alone misses.
- **13.6% critical failure rate.** 3 of 22 test molecules had the model's top-ranked solvent turn out to be insoluble, an honest failure mode, not hidden. (86.4% of the time the top pick does work, and 68.2% of the time the full ranking is exactly right.)
- **Possible selection-bias in per-solvent accuracy.** When a solvent shows higher held-out accuracy, is that because the model is genuinely better for that solvent, or simply because most of the molecules tested against it happen to be soluble (making the prediction easier by base rate)? Not yet disentangled — worth checking before presenting per-solvent accuracy as a measure of model quality.

### Changes to make

- [ ] Collect targeted experimental data for known DMSO-insoluble molecules to address the 297:1 class imbalance.
- [ ] Engineer peak-shape features (asymmetry, width) as the next lever past the accuracy ceiling.
- [ ] Check whether per-solvent accuracy differences reflect genuine model performance or just base-rate solubility differences across solvents.

Work in progress

## License

MIT — see [LICENSE](LICENSE).
