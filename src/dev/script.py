import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from data_converter.hplc_data_handler import solvent_physics_db, smiles_to_fingerprint









# Loading the trained material
model_path = os.path.join(current_dir, 'champion_xgb_model.joblib')
kbest_path = os.path.join(current_dir, 'k_best_selector.joblib')
scaler_path = os.path.join(current_dir, 'continuous_scaler.joblib')

model = joblib.load(model_path)
k_best = joblib.load(kbest_path)
scaler = joblib.load(scaler_path)

# Define default matrix size
num_mat_features = 7
default_matrix = np.zeros(num_mat_features)



# Prediction engine
def predict_nmr_solubility(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Error: Invalid SMILES string. Cannot calculate TPSA/MolWt."

    tpsa = Descriptors.TPSA(mol)
    molwt = Descriptors.MolWt(mol)
    
    raw_fp_array = smiles_to_fingerprint(smiles).reshape(1, -1)

    # Filter down to the 200 Best Bits using your saved selector
    fp_best = k_best.transform(raw_fp_array)

    results = []

    # Run thermodynamics loop with imported model and others
    for solvent, props in solvent_physics_db.items():
        continuous_raw = np.array([[
            tpsa, 
            molwt, 
            props['Dielectric'], 
            props['Hansen_D'], 
            props['Hansen_P'], 
            props['Hansen_H'],
            0.0, 
            0.0
        ]])
    
        # Scale continuous features with the saved math
        continuous_scaled = scaler.transform(continuous_raw)
        
        # Assemble & Predict
        X_final = np.hstack((fp_best, default_matrix.reshape(1, -1), continuous_scaled))
        prob_soluble = model.predict_proba(X_final)[0][1] 
        
        results.append({
            'Solvent': solvent,
            'Probability_Soluble (%)': round(prob_soluble * 100, 2)
        })

    return pd.DataFrame(results).sort_values(by='Probability_Soluble (%)', ascending=False).reset_index(drop=True)





# Execution block
if __name__ == "__main__":
    print("\n" + "="*50)
    print("AI Solubility Predictor ")
    print("="*50)
    
    while True:
        # Prompt the user for input
        user_smiles = input("\nEnter a SMILES string (or type 'q' to quit): ").strip()

        # X-RAY 1: See exactly what Python captured from your keyboard
        print(f"--> [DEBUG] Python received: '{user_smiles}'")
        
        # Check if they want to exit the program
        if user_smiles.lower() in ['q', 'quit', 'exit']:
            print("Shutting down...")
            break
            
        # Ignore accidental blank enters
        if not user_smiles:
            # X-RAY 2: See if it thinks you hit Enter on a blank line
            print("--> [DEBUG] Python thought the input was empty, looping back...")
            continue

        report = predict_nmr_solubility(user_smiles)

        if isinstance(report, str):
            print(report) # Prints the error message
        else:
            print("\n--- Final Solubility Report ---")
            print(report.to_string(index=False))




