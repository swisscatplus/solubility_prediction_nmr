import sys
import os
from rdkit import Chem

# This bit of code allows the Interface to "see" the Engine folder
# by looking at the parent directory (src)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# fetch the predictor function
from controller.predictor_runner import 'predictor_function'



def handle_user_request(solute_smi, solvent_name, temp=298.15):
    
   # This is the Interafce's main job: 
   # 1. Receive simple info
   # 2. Format it
   # 3. Send it to the Controller

    print(f"[Interface] Received request for {solute_smi} in {solvent_name}")
    
    # 1. Input Validation (Security check), ensures the input is actually text or not empty
    if not isinstance(solute_smi, str) or len(solute_smi) < 1:
        return {"error": "Invalid SMILES string"}
    
    # Chemical Validity Check
    mol = Chem.MolFromSmiles(solute_smi)
    if mol is None:
        return {"status": "error", "message": f"'{solute_smi}' is not a valid SMILES string"}

    # Sanity Check (is it too big?)
    if mol.GetNumAtoms() > 200:
        return {"status": "error", "message": "Molecule is too large for this specific model"}


    # 2. Call the Engine
    # We pass the heavy lifting to the other folder
    raw_result = "predictor_function"(solute_smi, solvent_name, temp)

    # 3. Output Control
    #decide exactly what the user sees, in this example confidence seems kind of uselss to be honest
    #(rounding the numbers to 3 decimal places)
    return {
        "status": "success",
        "prediction": round(raw_result['logS'], 3),
        "confidence": "High" if raw_result['uncertainty'] < 0.1 else "Low"
    }

