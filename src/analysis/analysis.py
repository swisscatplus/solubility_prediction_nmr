import pandas as pd


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

    