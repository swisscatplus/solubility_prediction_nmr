import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_dir)

os.environ["CATBOOST_DISABLE_JIT"] = "1"

import numpy as np
import pandas as pd
import joblib
import catboost
from data_converter.hplc_data_handler import solvent_physics_db

# Loading the trained material
model_path = os.path.join(current_dir, 'final_model.joblib')

model = joblib.load(model_path)



# Prediction engine
def rank_nmr_solubility(model, live_sensor_data, available_solvents, solvent_col='Solvent_Name'):
    universes = []

    for solvent in available_solvents:
        # Copy the base sensor data
        universe = live_sensor_data.copy()
        # Inject the parallel solvent
        universe[solvent_col] = solvent

        if solvent in solvent_physics_db:
            physics = solvent_physics_db[solvent]
            # (Make sure these keys exactly match how they are spelled in your hplc_data_handler!)
            universe['Sol_Dielectric'] = physics['Dielectric']
            universe['Sol_Hansen_D'] = physics['Hansen_D']
            universe['Sol_Hansen_P'] = physics['Hansen_P']
            universe['Sol_Hansen_H'] = physics['Hansen_H']
        else:
            print(f"WARNING: {solvent} skipped, missing from physics database!")
            continue
        
        universes.append(universe)

    df_simulation = pd.DataFrame(universes)
    expected_cols = model.feature_names_
    df_simulation = df_simulation[expected_cols]

    cat_cols = [model.feature_names_[i] for i in model.get_cat_feature_indices()] # that way if the model's categorical features ever change there won't be a problem
    pool = catboost.Pool(data=df_simulation, cat_features=cat_cols)
    probabilities = model.predict_proba(pool)[:, 1]

    df_simulation['Confidence'] = probabilities * 100
    ranked_results = df_simulation[[solvent_col, 'Confidence']].sort_values(by='Confidence', ascending=False)

    for rank, (index, row) in enumerate(ranked_results.iterrows(), 1):
        sol = row[solvent_col]
        conf = row['Confidence']
        
        if rank == 1:
            print(f"   {rank}. {sol.ljust(15)} : {conf:.2f}%")
        else:
            print(f"     {rank}. {sol.ljust(15)} : {conf:.2f}%")
    print("="*50)

    best_solvent = ranked_results.iloc[0][solvent_col]
    return best_solvent



# Execution block
if __name__ == "__main__":
    my_lab_solvents = ['MeOH','ACN','DMSO','DCM','CHCl3']

    live_peak_detected = {
        'MolWt': 581.26,
        'RT': 4.146,
        'Matrix ID Name': 'BlueBird 1',
        'Matrix_Vector_1': 3.54,
        'Matrix_Vector_2': 5.48,
        'Matrix_Vector_3': 5.2,
        'Matrix_Vector_4': 5.22,
        'Matrix_Vector_5': 5.68,
        'Matrix_Vector_6': 6.54,
        'Matrix_Vector_7': 0,
        'Matrix_Vector_8': 4.49,
        'Matrix_Vector_9': 6,
        'Matrix_Vector_10': 4.4
    }

    try:
        command = rank_nmr_solubility(
            model=model, 
            live_sensor_data=live_peak_detected, 
            available_solvents=my_lab_solvents
        )
        print(f"\n Best solvent: {command}")
    except KeyError as e:
        print(f"ERROR: Missing feature in sensor data — {e}")
    except Exception as e:
        print(f"ERROR: Prediction failed — {e}")