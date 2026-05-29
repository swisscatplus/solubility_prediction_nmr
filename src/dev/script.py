import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_dir)

import pandas as pd
import joblib

solvent_physics_db = {
    'MeOH': {'Dielectric': 32.7, 'Hansen_D': 15.1, 'Hansen_P': 12.3, 'Hansen_H': 22.3},
    'ACN': {'Dielectric': 37.5, 'Hansen_D': 15.3, 'Hansen_P': 18.0, 'Hansen_H': 6.1},
    'DMSO': {'Dielectric': 46.7, 'Hansen_D': 18.4, 'Hansen_P': 16.4, 'Hansen_H': 10.2},
    'DCM': {'Dielectric': 8.93, 'Hansen_D': 18.2, 'Hansen_P': 6.3, 'Hansen_H': 6.1},
    'CHCl3': {'Dielectric': 4.81, 'Hansen_D': 17.8, 'Hansen_P': 3.1, 'Hansen_H': 5.7}
}

# Loading the trained material and some configuration for next steps
LAB_SOLVENTS     = ['MeOH', 'ACN', 'DMSO', 'DCM', 'CHCl3']
model_path       = os.path.join(current_dir, 'final_model.joblib')


# Loading the csv file
def load_input(csv_path):
    df = pd.read_csv(csv_path)
    required = ['MolWt', 'RT', 'Matrix ID Name'] + [f'Matrix_Vector_{i}' for i in range(1, 11)]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df

# Prediction engine
def rank_single(model, sensor_row, row_idx):
    universes    = []
    solvent_names = []

    for solvent in LAB_SOLVENTS:
        universe = sensor_row.copy()

        if solvent in solvent_physics_db:
            physics = solvent_physics_db[solvent]
            universe['Sol_Dielectric'] = physics['Dielectric']
            universe['Sol_Hansen_D']   = physics['Hansen_D']
            universe['Sol_Hansen_P']   = physics['Hansen_P']
            universe['Sol_Hansen_H']   = physics['Hansen_H']
        else:
            print(f"WARNING: {solvent} skipped — missing from physics database!")
            continue

        universes.append(universe)
        solvent_names.append(solvent)   # track solvent name separately

    df_simulation = pd.DataFrame(universes)
    expected_cols = model.feature_names_in_
    df_simulation = df_simulation[expected_cols]
    probabilities = model.predict_proba(df_simulation)[:, 1]

    # Build ranked results using the separately tracked names
    ranked = pd.DataFrame({
        'Solvent':    solvent_names,
        'Confidence': probabilities * 100
    }).sort_values(by='Confidence', ascending=False)

    print(f"\n{'='*52}")
    print(f"  Unknown Molecule #{row_idx + 1}  "
          f"| MolWt: {sensor_row['MolWt']}  RT: {sensor_row['RT']}")
    print(f"{'='*52}")

    for rank, (_, row) in enumerate(ranked.iterrows(), 1):
        sol  = row['Solvent']
        conf = row['Confidence']
        marker = '▶' if rank == 1 else ' '
        print(f"  {marker}{rank}. {sol.ljust(10)} : {conf:.2f}%")

    print(f"{'='*52}")
    return ranked.iloc[0]['Solvent']


# Execution block
if __name__ == "__main__":
    model = joblib.load(model_path)

    csv_path = input("Enter the path to CSV file: ").strip()

    try:
        input_df = load_input(csv_path)
        print(f"\nLoaded {len(input_df)} molecule(s) from {csv_path}")

        results = []
        for idx, row in input_df.iterrows():
            try:
                best = rank_single(model, row.to_dict(), idx)
                results.append({'Molecule': idx + 1, 'Best_Solvent': best})
            except KeyError as e:
                print(f"ERROR on molecule {idx + 1}: missing feature {e}")
            except Exception as e:
                print(f"ERROR on molecule {idx + 1}: {e}")

        print(f"\n{'='*52}")
        print(f"  SUMMARY")
        print(f"{'='*52}")
        for r in results:
            print(f"  Molecule #{r['Molecule']:>3}  →  {r['Best_Solvent']}")
        print(f"{'='*52}\n")

    except ValueError as e:
        print(f"INPUT ERROR: {e}")
    except Exception as e:
        print(f"ERROR: {e}")