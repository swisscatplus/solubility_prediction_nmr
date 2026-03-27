import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from controller.predictor_runner import calculate_molecular_descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb


def count_over_under_estimation(compared_results_file, solvent_dict):
    df = pd.read_excel(compared_results_file)
    
    # Cleaning headers just in case
    df.columns = df.columns.str.strip()
    
    results_summary = {}

    print("--- Solubility Prediction Error Profile ---")
    
    for solvent in solvent_dict.keys():
        mag_col = f'Magnitude_{solvent}'
        coh_col = f'Coherence_{solvent}'
        
        if mag_col not in df.columns:
            continue
            
        mismatches = df[df[coh_col].astype(str).str.strip().str.lower() == 'false']
        
        # Now we count within those mismatches
        over_count = len(mismatches[mismatches[mag_col] > 0])
        under_count = len(mismatches[mismatches[mag_col] < 0])   
        # 3. Total Incoherent for context
        total_fail = over_count + under_count
        
        results_summary[solvent] = {
            "Overestimated": over_count,
            "Underestimated": under_count,
            "Total_Failures": total_fail
        }
        
        print(f"\nSolvent: {solvent}")
        print(f"  📈 Overestimated: {over_count}")
        print(f"  📉 Underestimated: {under_count}")
        print(f"  Total Mismatches: {total_fail}")

    return results_summary



def count_incoherence_molecule(compared_results_file, solvent_dict):
    """
    Counts how many molecules have 1, 2, 3... N incoherences across the solvent set.
    """
    # 1. Load the results
    df = pd.read_excel(compared_results_file)
    df.columns = df.columns.str.strip()

    # 2. Identify all the Coherence columns based on your dictionary
    coh_cols = [f'Coherence_{s}' for s in solvent_dict.keys()]

    # Filter to only use columns that actually exist in the file
    existing_cols = [col for col in coh_cols if col in df.columns]

    if not existing_cols:
        print("No Incoherence columns found. Check your file and solvent dictionary.")
        return
    
    # 3. Calculate the sum of 'False' per row
    incoherence_matrix = df[existing_cols].astype(str).apply(lambda x: x.str.strip() == 'False')

    # Sum across the rows (axis=1) to get the total number of errors per compound
    df['molecule_error_count'] = incoherence_matrix.sum(axis=1)

    # 4. Display the results
    print(f"--- Molecular Incoherence Profile ---")
    print(f"Total compounds analyzed: {len(df)}")

    # Loop from 1 to the total number of solvents in dictionary
    max_solvents = len(solvent_dict)
    for i in range(1, max_solvents + 1):
        num_molecules = (df['molecule_error_count'] == i).sum()
        print(f"Molecules with exactly {i} incoherence(s): {num_molecules}")

    # Adding a '0' count for context, to compare
    perfect_matches = (df['molecule_error_count'] == 0).sum()
    print(f"-------------------------------------")
    print(f"Molecules with 0 incoherences (Perfect match): {perfect_matches}")

    return df


 # this actually intakes the counting over/under estimation results given previously, so run it through the file first
def bar_plot_over_under_estimation(counted_results_file):
    # 1. Prepare data for Seaborn (Long format)
    data = []
    for solvent, stats in counted_results_file.items():
        data.append({'Solvent': solvent, 'Type': 'Overestimated', 'Count': stats['Overestimated']})
        data.append({'Solvent': solvent, 'Type': 'Underestimated', 'Count': stats['Underestimated']})
    
    df_plot = pd.DataFrame(data)

    sns.set_theme(style="ticks")
    
    # We assign the figure to a variable 'fig'
    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. Create Plot
    # Hue handles the grouping of Over/Under automatically
    sns.barplot(
        data=df_plot, 
        x='Solvent', 
        y='Count', 
        hue='Type', 
        palette={'Overestimated': '#ff9999', 'Underestimated': '#66b3ff'},
        ax=ax
        
    )
    
    plt.title('Solubility prediction errors for each solvent: Over vs Under estimation', fontsize=15)
    plt.ylabel('Number of Molecules', fontsize=12)
    plt.xlabel('Solvent', fontsize=12)

    plt.show()

    return fig

    

def butterfly_plot_over_under_estimation(counted_results_file):
    # Prepare data
    data = []
    for solvent, stats in counted_results_file.items():
        data.append({
            'Solvent': solvent, 
            'Over': stats['Overestimated'], 
            'Under': -stats['Underestimated']
        })
    
    df_plot = pd.DataFrame(data).sort_values(by='Solvent')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting
    sns.barplot(x='Over', y='Solvent', data=df_plot, color='#ff9999', label='Overestimated')
    sns.barplot(x='Under', y='Solvent', data=df_plot, color='#66b3ff', label='Underestimated')
    
    ax.axvline(0, color='black', lw=1.5)
    
   # 4. Fix x-axis labels - REMOVED the [0] index
    ticks = ax.get_xticks() # Get the full list/array of ticks
    ax.set_xticks(ticks)    # This prevents a warning in newer Matplotlib versions
    ax.set_xticklabels([int(abs(t)) for t in ticks])
    
    plt.title('Butterfly Plot: Error Distribution', fontsize=14, pad=20)
    plt.xlabel('Count (← Underestimated | Overestimated →)')
    plt.legend()
    
    plt.show()

    return fig


def plot_ordered_solubility_array(compared_results_file, solvent_name):
    """
    Creates an ordered array (Waterfall Plot) of unique molecules 
    sorted by predicted logS to visualize the model's dynamic range.
    """
    # 1. Load the compared data
    df = pd.read_excel(compared_results_file)
    
    logS_col = f'predicted_logS_{solvent_name}'
    binary_col = solvent_name # The 1/0 lab result column
    
    if logS_col not in df.columns:
        print(f"Error: {logS_col} not found in the file.")
        return

    # 2. Sort by logS (Highest -> Lowest)
    # This creates the 'Ordered Array' ranking
    df_sorted = df.sort_values(by=logS_col, ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = df_sorted.index + 1 

    # 3. Setup the visual style
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 4. Plotting the Rank vs logS
    # We use 'hue' to color points by their experimental outcome
    sns.scatterplot(
        data=df_sorted, 
        x='Rank', 
        y=logS_col, 
        hue=binary_col, 
        palette={1: '#2ecc71', 0: '#e74c3c'}, # Green for Soluble, Red for Insoluble
        s=30, 
        alpha=0.8,
        edgecolor='white',
        linewidth=0.5,
        ax=ax
    )

    # 5. Add the 0.05 mol/L Threshold Line
    # log10(0.05) is roughly -1.3
    threshold_log = np.log10(0.05)
    ax.axhline(threshold_log, color='black', linestyle='--', linewidth=1.5, label='0.05 mol/L Threshold')

    # 6. Final Polish
    plt.title(f'Ordered Solubility Array: {solvent_name}', fontsize=15, pad=15)
    plt.xlabel('Molecule Rank (Most Soluble → Least Soluble)', fontsize=12)
    plt.ylabel('Predicted logS', fontsize=12)
    
    # Customizing the legend for clarity
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Insoluble (1)', 'Soluble (0)', 'Threshold)'], title='Lab Result', loc='upper right')
    
    sns.despine()
    plt.tight_layout()
    plt.show()

    plt.savefig(f'data/solubility_arrays/solubility_array_{solvent_name}.png', dpi=300, bbox_inches='tight')

    return fig



def plot_rt_ordered_solubility(compared_results_file, solvent_name):
    """
    Orders molecules by Retention Time (RT) to see if prediction 
    errors correlate with molecular hydrophobicity.
    """
    # 1. Load data
    df = pd.read_excel(compared_results_file)
    
    logS_col = f'predicted_logS_{solvent_name}'
    binary_col = solvent_name
    rt_col = 'RT' # Assuming the column is exactly named 'RT'

    # Check if RT exists
    if rt_col not in df.columns:
        print(f"Error: '{rt_col}' column not found. Check your file headers!")
        print(f"Available: {df.columns.tolist()}")
        return
    
    # 2. Sort by Retention Time (Low RT -> High RT)
    # This aligns molecules from least hydrophobic to most hydrophobic
    df_sorted = df.sort_values(by=rt_col).reset_index(drop=True)
    
    # 3. Setup Plot
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 4. Scatter Plot: RT on X, logS on Y
    # Color by Experimental Result to catch the mismatches
    sns.scatterplot(
        data=df_sorted, 
        x=rt_col, 
        y=logS_col, 
        hue=binary_col, 
        palette={1: '#2ecc71', 0: '#e74c3c'}, 
        s=40, 
        alpha=0.7,
        edgecolor='w',
        ax=ax
    )

    # 5. Add the 0.05 Threshold Line
    threshold_log = np.log10(0.05)
    ax.axhline(threshold_log, color='black', linestyle='--', alpha=0.8, label='0.05 Threshold')

    # 6. Formatting
    plt.title(f'Solubility Predictions vs. Retention Time: {solvent_name}', fontsize=15)
    plt.xlabel('Retention Time (min) → Increasing Hydrophobicity', fontsize=12)
    plt.ylabel('Predicted logS', fontsize=12)
    
    # Legend cleanup
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [ 'Insoluble (0)', 'Soluble (1)', 'Threshold'], title='Lab Result')
    
    sns.despine()
    
    # Save to your data folder as requested
    plt.savefig(f'data/RT_trends/RT_trend_{solvent_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def plot_master_overlaid_multi_trends(compared_results_file, solvent_dict, reference_solvent):
    """
    Orders all molecules based on one 'Master' solvent's predicted logS, 
    then plots the logS for all solvents to compare trends.
    """
    df = pd.read_excel(compared_results_file)
    
    ref_col = f'predicted_logS_{reference_solvent}'
    if ref_col not in df.columns:
        print(f"Error: Reference solvent '{reference_solvent}' not found.")
        return

    # 1. Establish the "Master Order"
    # Sort by the reference solvent (Highest -> Lowest logS)
    df_sorted = df.sort_values(by=ref_col, ascending=False).reset_index(drop=True)
    df_sorted['Master_Rank'] = df_sorted.index + 1

    # 2. Reshape for Multi-Plotting (Long Format)
    plot_data = []
    for solvent in solvent_dict.keys():
        logS_col = f'predicted_logS_{solvent}'
        incoh_col = f'Coherence_{solvent}'
        
        if logS_col in df_sorted.columns:
            temp_df = df_sorted[['Master_Rank', logS_col, incoh_col]].copy()
            temp_df['Solvent_Label'] = solvent
            temp_df.columns = ['Rank', 'logS', 'Match', 'Solvent']
            plot_data.append(temp_df)

    full_df = pd.concat(plot_data)

    # 3. Create the Plot
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="ticks")

    # We use 'hue' for Solvent to see them all together
    # We use a scatter plot with a low alpha so we can see the density
    sns.scatterplot(
        data=full_df, 
        x='Rank', 
        y='logS', 
        hue='Solvent', 
        palette='viridis', # A nice distinct color scale
        s=20, 
        alpha=0.4, 
        edgecolor=None
    )

    # 4. Add the 0.05 mol/L Threshold Line
    threshold_log = np.log10(0.05)
    plt.axhline(threshold_log, color='black', linestyle='--', linewidth=2, label='0.05 Threshold')

    # 5. Formatting
    plt.title(f'Global Solubility Landscape (Ranked by {reference_solvent})', fontsize=16, pad=20)
    plt.xlabel(f'Molecules Ranked by {reference_solvent} Predicted Solubility', fontsize=12)
    plt.ylabel('Predicted logS', fontsize=12)
    
    # Move legend outside the plot so it doesn't cover the dots
    plt.legend(title='Solvents', bbox_to_anchor=(0.85, 1), loc='upper left')
    
    sns.despine()
    plt.grid(axis='y', linestyle=':', alpha=0.5)

    # Save to data folder
    save_name = f'data/overlaid_trends/Multi_Trend_Overlaid_by_{reference_solvent}.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()

    return full_df


def analyze_plot_chemical_bias(compared_results_file, solvent_name):
    df = pd.read_excel(compared_results_file)
    
    # Check if SMILES column exists
    if 'solute_smiles' not in df.columns:
        print("Error: 'solute_smiles' column missing. Cannot calculate descriptors.")
        return

    # 1. APPLY THE DESCRIPTOR FUNCTION
    # This creates the two new columns using the separate function
    print(f"--- Annotating {len(df)} molecules with LogP and MW ---")
    df[['LogP', 'MW']] = df['solute_smiles'].apply(
        lambda x: pd.Series(calculate_molecular_descriptors(x))
    )
    
    # 2. Setup the Plot
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="white")
    
    incoh_col = f'Coherence_{solvent_name}'
    
    # 3. Create a Scatter Plot of the Chemical Space
    # We use 'hue' to see where the Mismatches live
    sns.scatterplot(
        data=df, 
        x='LogP', 
        y=f'predicted_logS_{solvent_name}', 
        hue=incoh_col,
        palette={True: '#bdc3c7', False: '#e74c3c'}, # Grey for Correct, Red for Error
        alpha=0.7,
        s=60
    )
    
    # 4. Add "Trend" marginals (Density plots on the sides)
    # This shows if errors are skewed toward high LogP or high MW
    plt.title(f'Where does FastSolv fail in {solvent_name}?', fontsize=15)
    plt.xlabel('Hydrophobicity (LogP)', fontsize=12)
    plt.ylabel('Solubility (logS)', fontsize=12)
    
    sns.despine()
    plt.show()

    # 5. Print a quick summary for your TA
    mismatches = df[df[incoh_col] == 'False']
    print(f"\n--- Bias Analysis for {solvent_name} ---")

    plt.savefig(f'data/chemical_biases/chemical_bias_{solvent_name}.png', dpi=300, bbox_inches='tight')


# this function was recommended to be made to interpret the results of the overlaid graphs
def plot_solvent_correlation_heatmap(compared_results_file, solvent_dict):
    """
    Computes and plots the correlation between predicted logS values 
    across all solvents to check for model redundancy.
    """
    # 1. Load the data
    df = pd.read_excel(compared_results_file)
    
    # 2. Extract only the predicted logS columns
    # We use a dictionary comprehension to map the exact column names to just the solvent names for a cleaner plot
    logS_cols = {f'predicted_logS_{s}': s for s in solvent_dict.keys()}
    
    # Filter the dataframe to only include these columns
    subset_df = df[list(logS_cols.keys())].copy()
    
    # Rename the columns so the plot labels are just "MeOH", "ACN", etc.
    subset_df = subset_df.rename(columns=logS_cols)
    
    # 3. Calculate the Pearson Correlation Matrix
    corr_matrix = subset_df.corr()

    # 4. Setup the Plot
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    # 5. Plot the Heatmap
    # We use vmin=0.5 and vmax=1.0 because logS correlations are rarely negative 
    # and we want to highlight the differences at the high end.
    sns.heatmap(
        corr_matrix, 
        annot=True,          # Show the exact numbers
        fmt=".3f",           # 3 decimal places for precision
        cmap='coolwarm',     # Red for highly correlated, Blue for less
        vmin=0.8, vmax=1.0,  # Focus the color scale on high correlations
        square=True, 
        linewidths=1, 
        linecolor='white',
        cbar_kws={"shrink": .8, 'label': 'Pearson Correlation (R)'}
    )
    
    plt.title('Prediction Redundancy: Inter-Solvent logS Correlation', fontsize=16, pad=20)
    
    # 6. Save and Show
    save_path = 'data/Solvent_Correlation_Heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Correlation Heatmap saved to: {save_path}")
    
    plt.show()

    return corr_matrix


def plot_mw_error_distribution(compared_results_file, solvent_name):
    """
    Compares the Molecular Weight distribution of correct predictions 
    vs. incorrect predictions to identify size-based bias.
    """
    df = pd.read_excel(compared_results_file)
    incoh_col = f'Coherence_{solvent_name}'
    
    if incoh_col not in df.columns:
        print(f" Error: {incoh_col} not found in the file.")
        return

    # 1. Calculate MW if it's not already in the file
    if 'MW' not in df.columns:
        if 'solute_smiles' in df.columns:
            print("--- Calculating Molecular Weights from SMILES ---")
            df['MW'] = df['solute_smiles'].apply(
                lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)) if pd.notnull(x) and Chem.MolFromSmiles(x) else None
            )
        else:
            print(" Error: No 'MW' or 'solute_smiles' column found. Cannot calculate weight.")
            return

    # Drop missing values and standardize the Match column
    df = df.dropna(subset=['MW', incoh_col])
    df['Match'] = df[incoh_col].astype(str).str.upper()

    # 2. Setup the Plot
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="ticks")
    
    # Boxplot shows the median and the bulk of the distribution
    sns.boxplot(
        data=df, 
        x='Match', 
        y='MW', 
        palette={'TRUE': '#bdc3c7', 'FALSE': '#e74c3c'}, 
        showfliers=False, # We hide outliers here because the swarmplot handles them
        width=0.5
    )
    
    # Swarmplot overlays the actual individual molecules as dots
    sns.swarmplot(
        data=df, 
        x='Match', 
        y='MW', 
        color=".25", 
        alpha=0.6,
        size=4
    )
    
    # 3. Formatting
    plt.title(f'Size Bias in FastSolv: Molecular Weight Errors ({solvent_name})', fontsize=15)
    plt.xlabel('Did the model predict correctly?', fontsize=12)
    plt.ylabel('Molecular Weight (g/mol)', fontsize=12)
    sns.despine()
    
    # 4. Save and Show
    plt.savefig(f'data/MW_Error_Dist/MW_Error_Dist_{solvent_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Print the hard numbers for your report
    mean_true = df[df['Match'] == 'TRUE']['MW'].mean()
    mean_false = df[df['Match'] == 'FALSE']['MW'].mean()
    
    print(f"\n--- Statistical Summary for {solvent_name} ---")
    print(f"Average MW of Correct Predictions:   {mean_true:.1f} g/mol")
    print(f"Average MW of Incorrect Predictions: {mean_false:.1f} g/mol")
    print(f"Difference: {abs(mean_false - mean_true):.1f} g/mol")

    return df





def get_h_bonds(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            return hbd, hba
        return None, None



def plot_hydrogen_bonding_bias(compared_results_file, solvent_name):
    """
    Analyzes prediction errors based on the number of Hydrogen Bond 
    Donors (HBD) and Acceptors (HBA) in the molecules.
    """
    df = pd.read_excel(compared_results_file)
    incoh_col = f'Coherence_{solvent_name}'
    
    if incoh_col not in df.columns or 'solute_smiles' not in df.columns:
        print(f" Error: Missing {incoh_col} or SMILES column.")
        return

    # 1. Calculate HBD and HBA using RDKit (H-bond donors, Hbond-acceptors)
    print(f"--- Calculating HBD & HBA for {solvent_name} ---")
    

    df[['HBD', 'HBA']] = df['solute_smiles'].apply(lambda x: pd.Series(get_h_bonds(x)))
    
    # Drop missing values and standardize the Match column
    df = df.dropna(subset=['HBD', 'HBA', incoh_col])
    df['Match'] = df[incoh_col].astype(str).str.upper()

    # 2. Setup the Plot
    plt.figure(figsize=(9, 7))
    sns.set_theme(style="whitegrid")
    
    # We use a jittered scatter plot so integer coordinates don't perfectly overlap
    sns.stripplot(
        data=df, 
        x='HBD', 
        y='HBA', 
        hue='Match', 
        palette={'TRUE': '#bdc3c7', 'FALSE': '#e74c3c'}, 
        dodge=True,    # Separates the True and False dots slightly
        jitter=0.25,   # Shakes the dots so we can see density
        alpha=0.7, 
        size=6
    )
    
    # 3. Formatting
    plt.title(f'Stickiness Bias: Hydrogen Bonding Errors ({solvent_name})', fontsize=16, pad=15)
    plt.xlabel('Number of Hydrogen Bond Donors (HBD)', fontsize=12)
    plt.ylabel('Number of Hydrogen Bond Acceptors (HBA)', fontsize=12)
    
    # Legend cleanup
    plt.legend(title='Model Prediction Correct?', bbox_to_anchor=(0.90, 1), loc='upper left')
    sns.despine()
    
    # 4. Save and Show
    save_path = f'data/HBond_bias/HBond_Bias_{solvent_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # 5. Quick Stats
    mismatches = df[df['Match'] == 'FALSE']
    print(f"Average HBD of Errors: {mismatches['HBD'].mean():.1f}")
    print(f"Average HBA of Errors: {mismatches['HBA'].mean():.1f}")

    return df





# This next function is part of the model analysis trend i've been working on.
# It's supposed to intake the big dataframe generated to input in Randomforest and dynamically remove one parameter
# to test the accuracy of the model (and other things). It'll return a .txt file.
# Basically and automation of the files seen in this very folder and 'future proof'.
def run_ablation_study(mega_df, features_to_remove=None):
   
    # haden't even thougt of removing these two as options, i mean if we delete them there's no point of training the model i guess
    target_col = 'Soluble'
    group_col = 'SMILES'

    all_features = [col for col in mega_df.columns if col not in [target_col, group_col]]


    # the function intakes a feature to remove or several ones, it can also be interactive using the following code. i think this is useful 
    # if no loop was expected to run.
    if features_to_remove is None:
        print("========================================")
        print("  AVAILABLE FEATURES TO REMOVE:")
        print("========================================")
        for f in all_features:
            print(f"  - {f}")
        print("  - None (Keep all features)")
        print("========================================")
        
        
        user_input = input("Type the exact name of the feature to remove: ").strip()

        # Convert the text string into a clean list of features: it cuts at commas, strips away the spaces
        if user_input.lower() == 'none' or user_input == '':
            features_to_remove = [] # Empty list = remove nothing
        else:
            features_to_remove = [f.strip() for f in user_input.split(',')]

    # If the user bypassed the input and passed a single string directly in code
    elif isinstance(features_to_remove, str):
        if features_to_remove.lower() == 'none':
            features_to_remove = []
        else:
            features_to_remove = [features_to_remove]
    
    removed_str = ", ".join(features_to_remove) if features_to_remove else "None"
    print(f"\nRemoving [{removed_str}] from model...")


    feature_arrays = []

    for col in all_features:
        if col in features_to_remove:
            continue # This skips the one we are removing

        # this is a 'type of data identificator', because it needs to check what it's holding in order to correctly stack (the if code after)
        sample_val = mega_df[col].iloc[0]
        if isinstance(sample_val, (list, np.ndarray)):
        # If it's an array (like Fingerprints or One-Hot Solvents)
            feature_arrays.append(np.vstack(mega_df[col].values))
        else:
        # If it's a single number (like RT, logS)
            feature_arrays.append(mega_df[col].values.astype(float).reshape(-1, 1))
            
    # Smash everything together        
    X_dynamic = np.hstack(feature_arrays)
    y = mega_df[target_col].values.astype(int)
    groups = mega_df[group_col].values
    
    print(f"Matrix shape after removal: {X_dynamic.shape}")


    # Splitting and training part 
    seeds = [42, 93, 123, 2024, 777]
    results = []

    for seed in seeds:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        train_idx, temp_idx = next(gss1.split(X_dynamic, y, groups))
        X_train, y_train = X_dynamic[train_idx], y[train_idx]
        X_temp, y_temp = X_dynamic[temp_idx], y[temp_idx]
        groups_temp = groups[temp_idx]

        gss2 = GroupShuffleSplit(n_splits=1, test_size=(1/3), random_state=seed)
        val_idx, _ = next(gss2.split(X_temp, y_temp, groups_temp))
        X_val, y_val = X_temp[val_idx], y_temp[val_idx]
        
        
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_val_pred = rf.predict(X_val)
        y_train_pred = rf.predict(X_train)

        # Metrics
        acc = accuracy_score(y_val, y_val_pred)
        train_acc = accuracy_score(y_train, y_train_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, labels=[0, 1], zero_division=0)
        
        results.append({
            'Removed_Feature': removed_str,
            'Seed': seed,
            'Train_Accuracy': train_acc,
            'Accuracy': acc,
            'Precision': precision[1],
            'Recall': recall[1],
            'F1_Score': f1[1]
        })

    # Saving the results, writing them in the text file
    results_df = pd.DataFrame(results)
    
    if features_to_remove:
        safe_feature_name = "_and_".join(features_to_remove).replace(" ", "_")
    else:
        safe_feature_name = "None"

    txt_filename = f"ablation_results_no_{safe_feature_name}.txt"

    output_folder = "/Users/arthurbenard/Project 1B/src/analysis/model_analysis_results"
    full_filepath = os.path.join(output_folder, txt_filename)

    with open(full_filepath, 'w') as f:
        f.write(f"=== RESULTS WITH [{removed_str}] REMOVED ===\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n--- Summary Statistics ---\n")
        f.write(results_df.drop(columns=['Removed_Feature', 'Seed']).agg(['mean', 'std']).T.to_string())
        
    print(f"   -> Results saved to {full_filepath}\n")
    
    return results_df



# This function will combine all dataframes generated right before and plot them in a single graph. 
# The aim is to observe the drop in accuracy each time features are removed.

def plot_comparison(master_df):
    metrics_to_plot = ['Accuracy', 'Recall', 'F1_Score', 'Train_Accuracy']

    # .melt function is to turn metrics in long table so that Seaborn reads them more easily.
    melted_df = master_df.melt(
        id_vars=['Removed_Feature', 'Seed'], 
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Score'
    )

    # This part is to establish the order of the x-axis, 'None' will be first.
    unique_features = list(master_df['Removed_Feature'].unique())
    if 'None' in unique_features:
        unique_features.remove('None')
        order = ['None'] + unique_features # 'None' goes first, then the rest
    else:
        order = unique_features

    # The whole plotting code
    num_x_categories = len(order)
    dynamic_width = max(10, num_x_categories * 2.5)
    plt.figure(figsize=(dynamic_width, 6))

    sns.boxplot(
        data=melted_df,
        x='Removed_Feature',
        y='Score',
        hue='Metric',
        order=order,
        palette=['#4C72B0', '#55A868', '#C44E52','#FFFF00'] # Classic Blue, Green, Red
    )

    

    plt.title('Ablation Study: Feature Importance Impact', fontsize=16, fontweight='bold')
    plt.xlabel('Removed Feature(s)', fontsize=12, fontweight='bold')
    plt.ylabel('Metric Score', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.05)


    plt.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()


    output_folder = "/Users/arthurbenard/Project 1B/src/analysis/model_analysis_results"
    save_path = os.path.join(output_folder, 'Ablation_SuperPlot.png')


    plt.savefig(save_path, dpi=300)
    print(f"Plot successfully saved to {save_path}")

    plt.show()




# This function cuts the model training data differently
def run_data_fraction(mega_df, train_percent_of_total):
    # quick reality check if wrong input
    if train_percent_of_total > 1.0:
        raise ValueError("Training data must be less than 1.0 so we have leftovers to test on.")
    
    print(f"\nRunning Learning Curve Step: {train_percent_of_total * 100:.0f}% of total data...")

# This part builds the matrix and tells the model what to look at
    target_col = 'Soluble'
    group_col = 'SMILES'
    all_features = [col for col in mega_df.columns if col not in [target_col, group_col]]
    
    feature_arrays = []
    for col in all_features:
        sample_val = mega_df[col].iloc[0]
        if isinstance(sample_val, (list, np.ndarray)):
            feature_arrays.append(np.vstack(mega_df[col].values))
        else:
            feature_arrays.append(mega_df[col].values.astype(float).reshape(-1, 1))
            
    X_dynamic = np.hstack(feature_arrays)
    y = mega_df[target_col].values.astype(int)
    groups = mega_df[group_col].values

    seeds = [49, 90, 105, 1873, 1010]
    results = []

    for seed in seeds:
            gss_train = GroupShuffleSplit(n_splits=1, train_size=train_percent_of_total, random_state=seed)
            train_idx, remaining_idx = next(gss_train.split(X_dynamic, y, groups))
            
            X_train, y_train = X_dynamic[train_idx], y[train_idx]
            
            # This is whatever is left over (e.g., if train is 10%, this is 90%)
            X_remaining, y_remaining = X_dynamic[remaining_idx], y[remaining_idx]
            groups_remaining = groups[remaining_idx]

            # Split the leftovers 2:1. 
            # A test_size of 1/3 means 33.3% goes to Test, and 66.6% goes to Validation (a perfect 2:1 ratio)
            gss_val_test = GroupShuffleSplit(n_splits=1, test_size=(1/3), random_state=seed)
            val_idx, test_idx = next(gss_val_test.split(X_remaining, y_remaining, groups_remaining))
            
            X_val, y_val = X_remaining[val_idx], y_remaining[val_idx]
            X_test, y_test = X_remaining[test_idx], y_remaining[test_idx]

            # Train and Evaluate
            rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed, n_jobs=-1)
            rf.fit(X_train, y_train)
            
            # The following is a measure taken to tell the model to be at a certain minimum sure of his answer, instead of having cases were at
            # 51% he thinks it's fine.
            # 1. Define how confident the AI must be to flag a molecule as Soluble (1)
            # 0.70 means it must be 70% confident. 
            CUSTOM_THRESHOLD = 0.70 

            # 2. Ask the model for its raw confidence percentages instead of its final guess
            # [:, 1] grabs the probability specifically for the "Soluble" class
            y_train_probs = rf.predict_proba(X_train)[:, 1]
            y_val_probs = rf.predict_proba(X_val)[:, 1]
            y_test_probs = rf.predict_proba(X_test)[:, 1]

            # 3. Apply the strict threshold to generate the new 1s and 0s
            y_train_pred = (y_train_probs >= CUSTOM_THRESHOLD).astype(int)
            y_val_pred = (y_val_probs >= CUSTOM_THRESHOLD).astype(int)
            y_test_pred = (y_test_probs >= CUSTOM_THRESHOLD).astype(int)

            # Metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, labels=[0, 1], zero_division=0)
            
            results.append({
                'Train_Size_Pct': int(train_percent_of_total * 100),
                'Seed': seed,
                'Train_Accuracy': train_acc,
                'Val_Accuracy': val_acc,
                'Test_Accuracy': test_acc,
                'Precision': precision[1],
                'Recall': recall[1],
                'F1_Score': f1[1]
            })

    # 3. Save to Text File
    results_df = pd.DataFrame(results)
    txt_filename = f"dynamic_learning_curve_{int(train_percent_of_total * 100)}pct.txt"
    
    output_folder = "/Users/arthurbenard/Project 1B/src/analysis/learning_curve_results"
    full_filepath = os.path.join(output_folder, txt_filename)

    with open(full_filepath, 'w') as f:
        f.write(f"=== LEARNING CURVE: {int(train_percent_of_total * 100)}% TRAINING DATA ===\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n--- Summary Statistics ---\n")
        f.write(results_df.drop(columns=['Seed']).agg(['mean', 'std']).T.to_string())
        
    print(f"   -> Results saved to {full_filepath}")
    
    return results_df




# Learning Curve plotting function
def plot_learning_curve(master_lc_df):
    
    print("Generating Learning Curve Plot...")
    
    # Filter the metrics. 
    # For a classic learning curve, comparing Train vs. Test is the gold standard to spot overfitting!
    metrics_to_plot = ['Train_Accuracy', 'Val_Accuracy', 'Test_Accuracy']
    
    # The 'Melt' Trick
    melted_df = master_lc_df.melt(
        id_vars=['Train_Size_Pct', 'Seed'], 
        value_vars=metrics_to_plot,
        var_name='Metric',
        value_name='Score'
    )
        
    # Build the Plot Canvas
    plt.figure(figsize=(10, 6))
    
    # Draw the Curve, lineplot
    sns.lineplot(
        data=melted_df,
        x='Train_Size_Pct',
        y='Score',
        hue='Metric',
        marker='o', # Adds a little dot at each actual data point (10%, 20%, etc.)
        palette=['#4C72B0', '#55A868', '#C44E52'],
        linewidth=2.5
    )
    
    # Make it look professional
    plt.title('Learning Curve: Model Performance vs. Training Data Size', fontsize=16, fontweight='bold')
    plt.xlabel('Percentage of Total Data Used for Training (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy Score', fontsize=12, fontweight='bold')
    plt.ylim(0.4, 1.05) # Adjusted slightly so the curves fill the screen better
    
    # Set the X-axis to actually show 10, 20, 30... instead of weird decimals
    plt.xticks(master_lc_df['Train_Size_Pct'].unique()) 
    
    # Move the legend outside the plot
    plt.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add a grid for both axes this time to make reading the curve easier
    plt.grid(axis='both', linestyle='--', alpha=0.7) 
    plt.tight_layout()
    
    # Save it to your results folder!
    output_folder = "/Users/arthurbenard/Project 1B/src/analysis/learning_curve_results"
    save_path = os.path.join(output_folder, 'Learning_Curve_Plot.png')
    
    plt.savefig(save_path, dpi=300)
    print(f"-> Plot successfully saved to {save_path}")
    
    plt.show()



# Almost the exact same function as run_ablation_study, but uses XGBoost.
def run_ablation_xgboost(mega_df, features_to_remove=None):
   
    target_col = 'Soluble'
    group_col = 'SMILES'

    all_features = [col for col in mega_df.columns if col not in [target_col, group_col]]

    if features_to_remove is None:
        print("========================================")
        print("  AVAILABLE FEATURES TO REMOVE:")
        print("========================================")
        for f in all_features:
            print(f"  - {f}")
        print("  - None (Keep all features)")
        print("========================================")
        
        user_input = input("Type the exact name of the feature to remove: ").strip()

        if user_input.lower() == 'none' or user_input == '':
            features_to_remove = [] 
        else:
            features_to_remove = [f.strip() for f in user_input.split(',')]

    elif isinstance(features_to_remove, str):
        if features_to_remove.lower() == 'none':
            features_to_remove = []
        else:
            features_to_remove = [features_to_remove]
    
    removed_str = ", ".join(features_to_remove) if features_to_remove else "None"
    print(f"\nRemoving [{removed_str}] from model...")

    feature_arrays = []

    for col in all_features:
        if col in features_to_remove:
            continue 

        sample_val = mega_df[col].iloc[0]
        if isinstance(sample_val, (list, np.ndarray)):
            feature_arrays.append(np.vstack(mega_df[col].values))
        else:
            feature_arrays.append(mega_df[col].values.astype(float).reshape(-1, 1))
            
    # Smash everything together        
    X_dynamic = np.hstack(feature_arrays)
    y = mega_df[target_col].values.astype(int)
    groups = mega_df[group_col].values
    
    print(f"Matrix shape after removal: {X_dynamic.shape}")

    # Splitting and training part 
    seeds = [42, 93, 123, 2024, 777]
    results = []

    for seed in seeds:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        train_idx, temp_idx = next(gss1.split(X_dynamic, y, groups))
        X_train, y_train = X_dynamic[train_idx], y[train_idx]
        X_temp, y_temp = X_dynamic[temp_idx], y[temp_idx]
        groups_temp = groups[temp_idx]

        gss2 = GroupShuffleSplit(n_splits=1, test_size=(1/3), random_state=seed)
        val_idx, _ = next(gss2.split(X_temp, y_temp, groups_temp))
        X_val, y_val = X_temp[val_idx], y_temp[val_idx]
        
        # --- XGBOOST REPLACEMENT LOGIC ---
        # Calculate dynamic class weight to mimic RF's 'balanced'
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=seed,
            eval_metric='logloss',
            scale_pos_weight=scale_weight, # Handles class imbalance
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        y_val_pred = xgb_model.predict(X_val)
        y_train_pred = xgb_model.predict(X_train)
        # ---------------------------------

        # Metrics
        acc = accuracy_score(y_val, y_val_pred)
        train_acc = accuracy_score(y_train, y_train_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, labels=[0, 1], zero_division=0)
        
        results.append({
            'Removed_Feature': removed_str,
            'Seed': seed,
            'Train_Accuracy': train_acc,
            'Accuracy': acc,
            'Precision': precision[1],
            'Recall': recall[1],
            'F1_Score': f1[1]
        })

    # Saving the results
    results_df = pd.DataFrame(results)
    
    if features_to_remove:
        safe_feature_name = "_and_".join(features_to_remove).replace(" ", "_")
    else:
        safe_feature_name = "None"

    # Updated filename so it doesn't overwrite your Random Forest text files!
    txt_filename = f"ablation_XGBOOST_no_{safe_feature_name}.txt"

    output_folder = "/Users/arthurbenard/Project 1B/src/analysis/model_analysis_results"
    
    full_filepath = os.path.join(output_folder, txt_filename)

    with open(full_filepath, 'w') as f:
        f.write(f"=== XGBOOST RESULTS WITH [{removed_str}] REMOVED ===\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n--- Summary Statistics ---\n")
        f.write(results_df.drop(columns=['Removed_Feature', 'Seed']).agg(['mean', 'std']).T.to_string())
        
    print(f"   -> Results saved to {full_filepath}\n")
    return results_df