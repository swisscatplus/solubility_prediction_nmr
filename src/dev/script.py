import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.append(parent_dir)
import torch

# 3. THE MONKEY PATCH: We hijack PyTorch's internal loading mechanism
_original_load = torch.load
def _mac_safe_load(*args, **kwargs):
    # Force every single model to load on the CPU, no matter what the file says
    kwargs['map_location'] = torch.device('cpu')
    return _original_load(*args, **kwargs)

# Overwrite the original function with our hijacked version
torch.load = _mac_safe_load
import fastsolv
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
            print(f"WARNING: {solvent} missing from physics database!")
        
        universes.append(universe)

    df_simulation = pd.DataFrame(universes)
    expected_cols = model.feature_names_
    df_simulation = df_simulation[expected_cols]

    probabilities = model.predict_proba(df_simulation)[:, 1]

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
        'MolWt': 854.9,
        'RT': 7.1,
        'Matrix ID Name': 'UNKNOWN_GHOST_HARDWARE',
        'Matrix_Vector_1': 3.54,
        'Matrix_Vector_2': -5.48,
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
        print(f"\n Command sent: {command}")
    except NameError:
        print("\n WARNING: 'model' is not defined. Make sure you train and load the CatBoost model before running this script.")