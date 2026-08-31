import sys
import importlib
import sys
sys.stdout = open('output.txt', 'w')
sys.path.append("/Users/arthurbenard/Project 1B/src")

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
from controller.predictor_runner import  generate_ml_file_with_logs, add_logs_to_df






# first part is making the dataframe

cleaned_df=cleaning_array_by_cas('/Users/arthurbenard/Project 1B/data/Fichier final (RT+sol)_with_smiles.xlsx', cas_column_name='CAS ')

sol_matrix= pd.read_excel('/Users/arthurbenard/Project 1B/data/Master_Solubility_Matrix.xlsx')



final_working_df = add_logs_to_df(cleaned_df, sol_matrix, NMR_SOLVENTS)

print("Gathering data for all solvents...")
all_dfs_v3 = []

for sol in NMR_SOLVENTS.keys():
    # since we are using cleaned_df, we just look for 'MeOH', 'ACN', etc.
    if sol in cleaned_df.columns:
        df_temp = generate_ml_file_with_logs(final_working_df, sol, NMR_SOLVENTS, smiles_to_fingerprint)
        all_dfs_v3.append(df_temp)
        print(f"Added {sol}: {len(df_temp)} rows")
    else:
        print(f"Skipped {sol}: Column not found")

# sticking everything together
mega_df_v3 = pd.concat(all_dfs_v3, ignore_index=True)
print(f"DataFrame built! Total Rows: {len(mega_df_v3)}")


X_solute = np.vstack(mega_df_v3['Sample Fingerprint'].values)  
X_rt     = mega_df_v3['RT'].values.reshape(-1, 1)    
X_logs   = mega_df_v3['logS'].values.reshape(-1, 1)  # NEW reshaping like above line
X_solvent= np.vstack(mega_df_v3['Solvent'].values)

X_v3 = np.hstack((X_solute, X_rt, X_logs, X_solvent))
y_v3 = mega_df_v3['Soluble'].values
groups_v3 = mega_df_v3['SMILES'].values

print(f" Final matrix shape: {X_v3.shape}")





# this is an automation going through the different random states chosen in seeds dict and plotting results

seeds = [42, 93, 123, 2024, 777] # 5 different random states
results = []
all_cms = []


print("\nRunning 5-Fold Stability Test...")

for seed in seeds:
    # split data safely by Molecule using the current seed
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss1.split(X_v3, y_v3, groups_v3))

    X_train, y_train, groups_temp = X_v3[train_idx], y_v3[train_idx], groups_v3[temp_idx]
    X_temp, y_temp = X_v3[temp_idx], y_v3[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=(1/3), random_state=seed)
    val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups_temp))

    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    
    #  Train Model, the 'balanced' class tells the AI that mistakes are no longer created equal (it is prone to guess if it sees a pattern in Soluble and Insoluble)
    rf_mega = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed, n_jobs=-1)
    rf_mega.fit(X_train, y_train)
    y_val_pred = rf_mega.predict(X_val)

    #  Collect Metrics
    acc = accuracy_score(y_val, y_val_pred)
    # This safely pulls the metrics specifically for class 1 (Soluble)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, labels=[0, 1], zero_division=0)
    
    cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1])
    all_cms.append(cm)
    
    results.append({
        'Seed': seed,
        'Accuracy': acc,
        'Precision (Class 1)': precision[1],
        'Recall (Class 1)': recall[1],
        'F1 Score (Class 1)': f1[1]
    })




results_df = pd.DataFrame(results)

print("\n=========================================")
print("  5-RUN STABILITY TEST RESULTS")
print("=========================================")
print(results_df.to_string(index=False))

print("\n--- Summary Statistics ---")
summary_df = results_df.drop(columns=['Seed']).agg(['mean', 'std']).T
print(summary_df.to_string())




# Plot 1: The Average Confusion Matrix
avg_cm = np.mean(all_cms, axis=0)

plt.figure(figsize=(6, 4))
sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues', cbar=False)
plt.title('Averaged Validation Confusion Matrix (5 Runs)')
plt.xlabel('AI Predicted (0=No, 1=Yes)')
plt.ylabel('Actual Lab Result (0=No, 1=Yes)')
plt.tight_layout()
plt.savefig('Average_Confusion_Matrix.png') # Saves as image file
plt.close()

# Plot 2: Boxplots of the Metrics to show variance visually
# Recall = Of all the molecules that are truly soluble, how many did the AI successfully catch?
# F-1 Score = mathematical middle ground between Precision and Recall
plt.figure(figsize=(8, 5))
sns.boxplot(data=results_df.drop(columns=['Seed', 'Accuracy']), palette="Set2")
plt.title('Model Variance Across 5 Different Random Splits (Class 1)')
plt.ylabel('Score')
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig('Metrics_Variance_Boxplot.png') # Saves as image file
plt.close()

print("\nPlots successfully saved as 'Average_Confusion_Matrix.png' and 'Metrics_Variance_Boxplot.png'")


sys.stdout.close()