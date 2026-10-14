# Project 1B - Solubility Prediction NMR

Predicting which deuterated NMR solvents an organic molecule dissolves in, from its HPLC retention time and molecular weight, without fingerprints or other structural descriptors.

Developed as part of Project 1B within the [SwissCat+](https://swisscat.org/) research platform at EPFL.

## Motivation

Choosing a deuterated NMR solvent for a new compound is usually guesswork or trial and error, and deuterated solvents are expensive. This project asks: **can we rank the NMR solvents a molecule will dissolve in using only what the HPLC already measured?**

The inputs are the retention time (RT), the HPLC column it was measured on, the molecular weight and the physical properties of each solvent. No fingerprints and no structural descriptors are used, which keeps the prediction close to what is available in a real lab workflow before full characterization.

## Data

| | |
|---|---|
| Molecules | **194**: 151 from the original dataset + 43 from 4 plate experiments (exp1, exp2, exp3, expDMSO2) |
| Solvents | **5** deuterated NMR solvents: MeOH, ACN, DMSO, DCM, CHCl₃ |
| HPLC columns | **7** (RTs from the SwissCat+ database; for plate molecules missing from it, the RT and column written in the plate file) |
| Rows for Model E | 5,560 (one row per molecule × HPLC column × solvent) |
| Labels | **real experimental solubility** (soluble = 1, insoluble = 0) |

The plate molecules were chosen on purpose to be **insoluble in DMSO**, to fight the model's tendency to always suggest DMSO. Their labels come from the "Mon observation" column of the plate file when it is filled, otherwise from the vision model "prediction" (33 observed labels, 207 from the vision model). Plate compounds with neither a CAS nor an RT cannot be used (5 in expDMSO2).

## Model

**Model E** (Arthur's final model, `rt_prediction_continued.ipynb` on the `restructuring` branch), an XGBoost classifier with 16 features:

- `RT` and `MolWt` (molecular weight, computed from the SMILES);
- the solvent: dielectric constant and Hansen parameters D, P, H;
- the HPLC column: RTs of 10 standard compounds measured on it (`Matrix_Vector_1..10`, from `data/raw/hplc_matrix_ids.xlsx`), so that RTs from different columns can be compared.

Settings: 1000 trees, learning rate 0.01, depth 4, `subsample=0.8`, `colsample_bytree=0.5`, `colsample_bynode=0.9`, `gamma=0.8`, `min_child_weight=10`, `reg_lambda=5`, `reg_alpha=1`, `scale_pos_weight` to balance soluble/insoluble, `random_state=72`. Splits are grouped by molecule, so no molecule is ever both in training and test.

To rank solvents for one molecule, the predicted probabilities are averaged over its HPLC columns and the solvents are sorted from most to least likely.

### Performance (notebook 04: 5-fold cross-validation × 3 repeats, 194 molecules)

| Metric | Model E | Simple rule |
|---|---|---|
| Correct first solvent (top-1) | **78 %** ± 5 | 71 % (always DMSO) |
| Working solvent in the first 2 suggestions (top-2)¹ | 94 % | 97 % (fixed order CHCl₃ > DMSO > DCM > MeOH > ACN) |
| Balanced accuracy (row level) | 62 % | 50 % (random) |
| Accuracy (row level) | 65 % | 69 % (always soluble) |

¹ over the molecules soluble in at least one solvent; top-1 on those is 85 %.

Without the plates, the original single split reproduces Arthur's numbers exactly (accuracy 0.7257, precision 0.8221 on the test set).

### What the analysis notebook shows (`05_analyse`)

- **Feature importance:** molecular weight > RT > solvent properties; the 10 column standards add almost nothing.
- **Calibration:** low probabilities are too pessimistic (announced 15-30 %, actually about 45 % soluble); Brier score 0.222 vs 0.221 for a constant. Recalibrate before using the probabilities in a cost function.
- **Errors:** error rate 34 % on the original labels, 35 % on vision-model labels, 47 % on labels from your own observation (31 pairs).
- **Effect of the plates:** top-1 on DMSO-insoluble molecules rises from 32 % (original data) to 39 % (+ 4 plates); overall top-1 stays at 78 %.
- **Learning curve:** balanced accuracy still rises with more molecules (58 % → 64 %); top-1 levels off around 78 %.
- **Solvents that behave alike:** DCM and CHCl₃ agree on 91 % of molecules, MeOH and ACN on 84 %. When the model puts CHCl₃ first it puts DCM second, which is why the fixed order wins at top-2.

## Workflow

1. **Add a plate.** Put the plate file (`*_output.xlsx`, sheet `Resultats`) in `data/raw/experiments/`.
2. **Rebuild the datasets.**
   ```bash
   python scripts/data_prep/add_new_molecules.py          # add --no-pubchem to work offline
   ```
   It fills the plate labels into the training tables, takes the RTs from the SwissCat+ HPLC database (or, for molecules that are not in it, the RT and column written in the plate file; method 1 when the method is not given), finds SMILES (PubChem by CAS, then by name; `data/raw/manual_smiles.csv` for manual fixes) and writes `data/interim/model_e_dataset.xlsx`.
3. **Run the notebooks, in order:**

   | Notebook | What it does |
   |---|---|
   | `01_split_data` | builds the Model E features and the fixed train/test split (`data/processed/train.csv`, `test.csv`) |
   | `02_train_model` | trains Model E and saves `models/xgb_model_e_16features.joblib` (`TRAIN_ON_ALL = True` for the final model on all data) |
   | `03_evaluate_model` | scores on the test set: row level and solvent choice per molecule |
   | `04_robustness_check` | the cross-validation (the only one): score on all molecules vs simple rules; saves the predictions to `data/processed/cv_predictions.csv` |
   | `05_analyse` (after 04) | data analysis (Venn, histograms, solvent similarity) and model analysis (learning curve, calibration, top-k, feature importance, errors, effect of each plate) |

4. **Predict new molecules.**
   ```bash
   python scripts/predict.py --input examples/example_input.csv
   ```
   The input needs `MolWt`, `RT`, `Matrix ID Name` and `Matrix_Vector_1..10` (see `examples/example_input.csv`).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In VS Code, select `.venv` as the Jupyter kernel. `fastsolv` (which pulls in PyTorch) is only needed for `src/main.py` and `scripts/dmso_search/`.

## Repository structure

Every path is defined once in `src/config.py`. Scripts and notebooks import it, so the project runs on any machine without editing paths. Run scripts from the project root.

```
.
├── data/
│   ├── raw/                 source files, never modified
│   │   ├── hplc_rt_solubility.xlsx        RT + experimental solubility (original dataset)
│   │   ├── swisscat_hplc_database.csv     full SwissCat+ HPLC database
│   │   ├── hplc_matrix_ids.xlsx           RTs of the 10 standards on each HPLC column
│   │   ├── experiments/                   plate experiments (*_output.xlsx)
│   │   ├── manual_smiles.csv              optional manual SMILES (CAS,SMILES)
│   │   └── epfl_inventory.csv             EPFL chemical inventory (not on git)
│   ├── interim/             generated tables: model_e_dataset.xlsx, molecules_unique_updated.xlsx, SMILES, fastsolv logS…
│   ├── processed/           train.csv, test.csv
│   └── archive/             old copies of the inventory (not on git)
│
├── models/
│   ├── xgb_model_e_16features.joblib     Model E, used by scripts/predict.py
│   └── archive/catboost_logs/            CatBoost training logs (abandoned model)
│
├── notebooks/               01_split_data → 02_train_model → 03_evaluate_model → 04_robustness_check → 05_analyse
│
├── src/                     reusable code (library)
│   ├── config.py            all project paths
│   ├── plot_style.py        one figure style for notebooks 04 and 05 (slide-ready)
│   ├── main.py              fastsolv analysis pipeline
│   ├── controller/          feature generation, training helpers
│   ├── data_converter/      HPLC data cleaning, SMILES, descriptors, solvent properties
│   ├── analysis/            plots, ablations, learning curves
│   └── interface/           SMILES input interface (prototype)
│
├── scripts/                 runnable entry points
│   ├── predict.py           rank NMR solvents for new molecules
│   ├── data_prep/           add_new_molecules.py, fetch_smiles_from_cas.py
│   ├── dmso_search/         find_dmso_insoluble.py (candidates from the EPFL inventory)
│   └── experiments/         random-forest experiments (fingerprint / RT / logS)
│
├── results/
│   ├── figures/             eda/data_analysis/, model_analysis/, experiments/, learning_curves/, ablation/, fastsolv_errors/
│   ├── reports/             text outputs
│   └── dmso_search/         top-50 DMSO-insoluble candidates + structure images
│
├── examples/example_input.csv    input format for scripts/predict.py
├── sandbox/                 debug dumps (not on git)
├── requirements.txt
└── LICENSE
```

## Limitations

- **DMSO bias.** DMSO dissolves most molecules, so the model suggests it first for about half of them. The plates targeting DMSO-insoluble molecules help (32 % → 39 % top-1 on those), but more are needed.
- **Row-level accuracy is below "always soluble".** The model is useful for ranking solvents, not for a yes/no answer per solvent.
- **Calibration.** Predicted probabilities are not reliable as absolute values yet.
- **Vision-model labels.** Most plate labels come from the vision model; the error analysis lists the pairs worth re-checking by eye.

## Next steps

- [ ] Cost function: rank solvents by expected cost using deuterated solvent prices and calibrated probabilities.
- [ ] Group solvents that behave alike (DCM/CHCl₃, MeOH/ACN) and use it in the model.
- [ ] Recalibrate the probabilities (e.g. isotonic or Platt scaling).
- [ ] More plates of DMSO-insoluble molecules (the learning curve is still rising).
- [ ] Peak-shape features (asymmetry, width) as new information beyond RT.

## License

MIT, see [LICENSE](LICENSE).
