import pandas as pd
import numpy as np
from usearch.index import Index
from tqdm import tqdm
from rdkit import Chem
from chemlactica.mol_opt.utils import get_morgan_fingerprint
from rdkit.Chem.QED import qed


def get_fing_from_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return np.packbits(np.array(get_morgan_fingerprint(mol)))
    except Exception as e:
        print(f"failed to parser: {smiles}, {e}")
        return None


def main():
    index = Index.restore("/nfs/dgx/raid/chem/vector_db/pubchem/index-ecfp2-qed>0.9.usearch")

    molecules_df = pd.read_csv("data/zinc/inference_inputs/qed_test_w_vals.csv")
    del molecules_df["Unnamed: 0"]
    lead_molecules = molecules_df.smiles.to_list()

    lead_molecules_finerprints = np.array([get_fing_from_smiles(s) for s in lead_molecules])
    print(lead_molecules_finerprints.shape)
    # distances = []
    # for fing in tqdm(lead_molecules_finerprints):
    #     distances.append(index.search(fing, count=1, exact=True, log=False, threads=1).distances[0])
    # distances = np.array(distances)
    distances = index.search(lead_molecules_finerprints, count=1, exact=True, log=True, threads=16).distances
    print(distances.shape)
    print(f"Number of found: {np.sum(distances <= 0.6)} / {len(lead_molecules)}")
    print(f"Success rate: {np.sum(distances <= 0.6) / len(lead_molecules):.3f}%")


if __name__ == "__main__":
    main()