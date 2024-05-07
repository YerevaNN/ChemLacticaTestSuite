import math
from pathlib import Path
import random
from rdkit.Chem.QED import qed

def generative_yield(mol_buffer,threshold):
    num_satisfactory_mols = 0
    for molecule_entry in mol_buffer:
        score = mol_buffer[mol_entry.smiles][0]
        if molecule_entry.score > threshold:
            num_satisfactory_mols += 1
    return num_satisfactory_mols

def oracle_burden(mol_buffer,n_molecules_needed,current_num_oracle_calls,threshold):
    num_satisfactory_mols = 0
    for molecule_entry in mol_buffer:
        score = mol_buffer[mol_entry.smiles][0]
        if molecule_entry.score > threshold:
            num_satisfactory_mols += 1
    if num_satisfactory_mols >= n_molecules_needed:
        return current_num_oracle_calls
    else:
        return 0

def vina_prompts_post_processor(prompt):
    # the prompt starts with having the similar molecules in int
    weight_val = round(random.uniform(250, 500), 2)
    weight_string = "{:.2f}".format(weight_val)

    qed_val = round(random.uniform(0.85, 0.95), 2)
    qed_string = "{:.2f}".format(weight_val)

    prompt += f"[WEIGHT]{weight_string}[/WEIGHT]"
    prompt += f"[QED]{qed_string}[/QED]"

    return prompt



