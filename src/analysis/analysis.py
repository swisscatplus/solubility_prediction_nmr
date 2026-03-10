import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from controller.predictor_runner import calculate_molecular_descriptors


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
        y='MW', 
        hue=incoh_col,
        palette={True: '#bdc3c7', False: '#e74c3c'}, # Grey for Correct, Red for Error
        alpha=0.7,
        s=60
    )
    
    # 4. Add "Trend" marginals (Density plots on the sides)
    # This shows if errors are skewed toward high LogP or high MW
    plt.title(f'Where does FastSolv fail in {solvent_name}?', fontsize=15)
    plt.xlabel('Hydrophobicity (LogP)', fontsize=12)
    plt.ylabel('Molecular Weight (g/mol)', fontsize=12)
    
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