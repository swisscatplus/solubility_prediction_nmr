import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
        print("❌ No Incoherence columns found. Check your file and solvent dictionary.")
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

