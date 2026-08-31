import sys
import importlib
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

src_path = os.path.abspath(os.path.join(current_dir, '..'))

if src_path not in sys.path:
    sys.path.append(src_path)

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

# Force Jupyter to reload the module from the hard drive
import data_converter.hplc_data_handler
importlib.reload(data_converter.hplc_data_handler)

# Now do the specific imports
from data_converter.hplc_data_handler import cleaning_array_by_cas, NMR_SOLVENTS, smiles_to_fingerprint
from controller.predictor_runner import generate_ml_ready_file_fingerprint_solvent









# Redirect output for this experiment
sys.stdout = open('output_solv_fp.txt', 'w')



cleaned_df=cleaning_array_by_cas('/Users/arthurbenard/Project 1B/data/Fichier final (RT+sol)_with_smiles.xlsx', cas_column_name='CAS ')

print("Gathering data ...")
all_dfs_v2 = []

for sol in NMR_SOLVENTS.keys():
    if sol in cleaned_df.columns:
        # using the function that EXCLUDES logS
        df_temp = generate_ml_ready_file_fingerprint_solvent(cleaned_df, sol, NMR_SOLVENTS, smiles_to_fingerprint)
        all_dfs_v2.append(df_temp)
        print(f"Added {sol}: {len(df_temp)} rows")

mega_df_v2 = pd.concat(all_dfs_v2, ignore_index=True)





X_solute = np.vstack(mega_df_v2['Sample Fingerprint'].values)  
X_rt     = mega_df_v2['RT'].values.reshape(-1, 1)    
X_solvent= np.vstack(mega_df_v2['Solvent'].values)


X_v2 = np.hstack((X_solute, X_rt, X_solvent)) 
y_v2 = mega_df_v2['Soluble'].values
groups_v2 = mega_df_v2['SMILES'].values

print(f"Matrix shape (with solvent fingerprint): {X_v2.shape}") # should be 2054 columns



seeds = [42, 93, 123, 2024, 777]
results_v2 = []
all_cms_v2 = []

for seed in seeds:
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss1.split(X_v2, y_v2, groups_v2))
    X_train, y_train = X_v2[train_idx], y_v2[train_idx]
    X_temp, y_temp = X_v2[temp_idx], y_v2[temp_idx]
    groups_temp = groups_v2[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=(1/3), random_state=seed)
    val_idx, _ = next(gss2.split(X_temp, y_temp, groups_temp))
    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    

    # Train
    rf_v2 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed, n_jobs=-1)
    rf_v2.fit(X_train, y_train)
    y_val_pred = rf_v2.predict(X_val)




    # Metrics
    acc = accuracy_score(y_val, y_val_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, labels=[0, 1], zero_division=0)
    
    results_v2.append({
        'Seed': seed,
        'Accuracy': acc,
        'Precision (Class 1)': precision[1],
        'Recall (Class 1)': recall[1],
        'F1 Score (Class 1)': f1[1]
    })
    all_cms_v2.append(confusion_matrix(y_val, y_val_pred, labels=[0, 1]))






results_df_v2 = pd.DataFrame(results_v2)
print("\n--- RESULTS Solvent FP ---")
print(results_df_v2.to_string(index=False))
print("\nSummary Statistics:")
print(results_df_v2.drop(columns=['Seed']).agg(['mean', 'std']).T.to_string())


avg_cm_v2 = np.mean(all_cms_v2, axis=0)
plt.figure(figsize=(6, 4))
sns.heatmap(avg_cm_v2, annot=True, fmt='.1f', cmap='Reds', cbar=False)
plt.title('Avg Confusion Matrix')
plt.savefig('Confusion_Matrix_Solvent_FP.png')
plt.close()

# Close output file
sys.stdout.close()