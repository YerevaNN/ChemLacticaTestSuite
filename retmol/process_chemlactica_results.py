from pathlib import Path
from rdkit.Chem.QED import qed
from rdkit import Chem
from chemlactica.mol_opt.utils import tanimoto_dist_func, get_morgan_fingerprint
import pandas as pd
import argparse
import os
from tqdm import tqdm


def get_line(line, s, e='\n'):
    if line.find(s) != -1 and line.find(e) != -1:
        return line[line.find(s) + len(s): line.find(e)]
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    lead_mols_df = pd.read_csv("../data/zinc/inference_inputs/qed_test_w_vals.csv")
    lead_smiles_list = lead_mols_df.smiles.to_list()

    total = 0
    num_optim = 0
    for i in tqdm(range(len(lead_smiles_list))):
        log_file_path = os.path.join(args.path, f"mol-{i}-2.log")
        if not os.path.exists(log_file_path):
            break
        with open(log_file_path, "r") as _f:
            all_lines = _f.readlines()
        lead_smiles = lead_smiles_list[i]
        lead_mol = Chem.MolFromSmiles(lead_smiles)
        for line in all_lines:
            smiles = get_line(line, "generated smiles: ", ", score:")
            if smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    qed_score = qed(mol)
                    sim_to_lead = tanimoto_dist_func(
                        get_morgan_fingerprint(mol), get_morgan_fingerprint(lead_mol)
                    )
                    if qed_score >= 0.9 and sim_to_lead >= 0.4:
                        num_optim += 1
                        break
                except Exception as e:
                    print(e)
                    continue
        total += 1
    print(num_optim, total)
    print(f"Success rate {num_optim / total:.4f}")