import pandas as pd 
import pubchempy as pcp
import time
import csv
import os

def main():
    #import du CSV 
    df = pd.read_csv('~/workspace/ProjectII/solubility_prediction_nmr/data/containers.csv', encoding='latin-1')

    #see column 
    print(df.columns.tolist())

    #transforme en string et enlève les espaces avant et après pour pas qu'il mettent des faux positifs
    df["CAS #"] = df["CAS #"].astype(str).str.strip()
    #on met en string pour pouvoir comparer avec le format dans pubchem

    #garde que les lignes NaN, vides, cases NaN
    #notNa retourne true or false - true si la valeur existe
    df_molecules = df[df["CAS #"].notna() & (df["CAS #"] != "") & (df["CAS #"].str.lower() != "nan")].copy()

    print(f"Nbr of line before : {len(df)}")
    print(f"Nbr of line after : {len(df_molecules)}")

    #importer depuis pubchem
    #??
    def get_smiles_from_cas(cas_number):
        """recover SMILES from CAS."""
        try:
            compounds = pcp.get_compounds(cas_number, 'name')  # CAS traité comme un nom/identifiant
            if compounds:
                return compounds[0].canonical_smiles
            else:
                return None
        except Exception as e:
            print(f"Error for CAS {cas_number} : {e}")
            return None
        
    progress_file = os.path.expanduser("~/workspace/ProjectII/solubility_prediction_nmr/data/smiles_progress_temp.csv")

    if os.path.exists(progress_file):
        smiles_list = pd.read_csv(progress_file)["SMILES"].tolist()
        print(f"Reprise : {len(smiles_list)} CAS déjà traités.")
    else:
        smiles_list = []

    start_index = len(smiles_list)  # on continue là où on s'est arrêté

    for i in range(start_index, len(df_molecules)):
        cas = df_molecules["CAS #"].iloc[i]
        smiles = get_smiles_from_cas(cas)
        smiles_list.append(smiles)
        time.sleep(1)  # pour ne pas surcharger l'API PubChem

        # Sauvegarde intermédiaire tous les 100 CAS
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(df_molecules)} traités...")
            pd.DataFrame({"SMILES": smiles_list}).to_csv(progress_file, index=False)

    # Sauvegarde finale de la progression (au cas où le dernier lot < 100 n'a pas été sauvé)
    pd.DataFrame({"SMILES": smiles_list}).to_csv(progress_file, index=False)

    # Ajoute la colonne SMILES au DataFrame final
    df_molecules["SMILES"] = smiles_list

    # Sauvegarde du fichier final complet
    output_path = os.path.expanduser("~/workspace/ProjectII/solubility_prediction_nmr/data/containers_with_smiles.csv")
    df_molecules.to_csv(output_path, index=False)
    print(f"Fichier final sauvegardé : {output_path}")

if __name__ == "__main__":
    main()