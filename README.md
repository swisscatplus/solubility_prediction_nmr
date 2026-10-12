# Project 1B - Solubility Prediction NMR

Predicting which deuterated NMR solvents an organic molecule dissolves in, from its HPLC retention time and molecular weight, without fingerprints or other structural descriptors.

Developed as part of Project 1B within the [SwissCat+](https://swisscat.org/) research platform at EPFL.

## Motivation

Choosing a deuterated NMR solvent for a new compound is usually guesswork or trial and error, and deuterated solvents are expensive. This project asks: **can we rank the NMR solvents a molecule will dissolve in using only what the HPLC already measured?**

The inputs are the retention time (RT), the HPLC column it was measured on, the molecular weight and the physical properties of each solvent. No fingerprints and no structural descriptors are used, which keeps the prediction close to what is available in a real lab workflow before full characterization.

## Data

| | |
|---|---|
| Molecules | **187**: 151 from the original dataset + 36 from 3 plate experiments (exp1, exp2, exp3) |
| Solvents | **5** deuterated NMR solvents: MeOH, ACN, DMSO, DCM, CHCl₃ |
| HPLC columns | **7** (BlueBird 1 preferred; another column is used when BlueBird 1 has no signal) |
| Rows for Model E | 5,525 (one row per molecule × HPLC column × solvent) |
| Labels | **real experimental solubility** (soluble = 1, insoluble = 0) |

The plate molecules were chosen on purpose to be **insoluble in DMSO**, to fight the model's tendency to always suggest DMSO. Their labels come from the "Mon observation" column of the plate file when it is filled, otherwise from the vision model "prediction" (27 observed labels, 153 from the vision model).

## Model

**Model E** (Arthur's final model, `rt_prediction_continued.ipynb` on the `restructuring` branch), an XGBoost classifier with 16 features:

- `RT` and `MolWt` (molecular weight, computed from the SMILES);
- the solvent: dielectric constant and Hansen parameters D, P, H;
- the HPLC column: RTs of 10 standard compounds measured on it (`Matrix_Vector_1..10`, from `data/raw/hplc_matrix_ids.xlsx`), so that RTs from different columns can be compared.

Settings: 1000 trees, learning rate 0.01, depth 4, `subsample=0.8`, `colsample_bytree=0.5`, `colsample_bynode=0.9`, `gamma=0.8`, `min_child_weight=10`, `reg_lambda=5`, `reg_alpha=1`, `scale_pos_weight` to balance soluble/insoluble, `random_state=72`. Splits are grouped by molecule, so no molecule is ever both in training and test.

To rank solvents for one molecule, the predicted probabilities are averaged over its HPLC columns and the solvents are sorted from most to least likely.

### Performance (5-fold cross-validation × 3 repeats, 187 molecules)

| Metric | Model E | Simple rule |
|---|---|---|
| Correct first solvent (top-1) | **80 %** | 73 % (always DMSO) |
| Working solvent in the first 2 suggestions (top-2)¹ | 94 % | 97 % (fixed order DMSO > CHCl₃ > DCM > MeOH > ACN) |
| Balanced accuracy (row level) | 62 % | 50 % (random) |
| Accuracy (row level) | 65 % | 68 % (always soluble) |

¹ over the molecules soluble in at least one solvent; top-1 on those is 86 %.

Without the plates, the original single split reproduces Arthur's numbers exactly (accuracy 0.7257, precision 0.8221 on the test set).

### What the analysis notebook shows (`05_analyse`)

- **Feature importance:** molecular weight > RT > solvent properties; the 10 column standards add almost nothing.
- **Calibration:** probabilities are too pessimistic (Brier score 0.227 vs 0.217 for a constant). They should be recalibrated before being used in a cost function.
- **Errors:** labels from your own observation are mispredicted 63 % of the time, vision-model labels 40 %, original data 33 %. The plate molecules are the hard cases, as intended.
- **Effect of the plates:** top-1 on DMSO-insoluble molecules rises from 33 % (original data) to 41 % (+ 3 plates), overall top-1 stays at 80 %.
- **Learning curve:** Model E is still improving with more molecules, so more plates should help.
- **Solvents that behave alike:** DCM and CHCl₃ agree on 91 % of molecules, MeOH and ACN on 84 %.

## Workflow

1. **Add a plate.** Put the plate file (`*_output.xlsx`, sheet `Resultats`) in `data/raw/experiments/`.
2. **Rebuild the datasets.**
   ```bash
   python scripts/data_prep/add_new_molecules.py          # add --no-pubchem to work offline
   ```
   It fills the plate labels into the training tables, takes the BlueBird 1 RT (or another column when BlueBird 1 has no signal), finds SMILES (PubChem by CAS, then by name; `data/raw/manual_smiles.csv` for manual fixes) and writes `data/interim/model_e_dataset.xlsx`.
3. **Run the notebooks, in order:**

   | Notebook | What it does |
   |---|---|
   | `01_split_data` | builds the Model E features and the fixed train/test split (`data/processed/train.csv`, `test.csv`) |
   | `02_train_model` | trains Model E and saves `models/xgb_model_e_16features.joblib` (`TRAIN_ON_ALL = True` for the final model on all data) |
   | `03_evaluate_model` | scores on the test set: row level and solvent choice per molecule |
   | `04_robustness_check` | cross-validation on all molecules, with simple-rule baselines |
   | `05_analyse` | data analysis (Venn, histograms, solvent similarity) and model analysis (learning curve, calibration, top-k, feature importance, errors, effect of each plate) |

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

- **DMSO bias.** DMSO dissolves most molecules, so the model suggests it first for about half of them. The plates targeting DMSO-insoluble molecules help (33 % → 41 % top-1 on those), but more are needed.
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
