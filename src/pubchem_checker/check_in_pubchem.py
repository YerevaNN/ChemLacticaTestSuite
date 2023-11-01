from tqdm import tqdm
from stringzilla import Str, Strs, File
from rdkit import Chem
from rdkit import RDLogger
from typing import List, Dict
from pprint import pprint

RDLogger.DisableLog('rdApp.*')


def check_in_pubchem(molecule_list: List[str]):
    # cid_smiles = Str(File('CID-SMILES')).split(separator="\n")
    cid_inchi = Str(File('/home/tigranfahradyan/novelty_checker/inchi'))
    found_molecules: Dict[str, int] = {smiles: False for smiles in molecule_list}
    for smiles_str in tqdm(molecule_list):
        # cid, smiles = cid_smiles[i].split("\t")
        # smiles_str = "".join([c for c in smiles])
        try:
            inchi = Chem.MolToInchi(Chem.MolFromSmiles(smiles_str))
            start = cid_inchi.find(inchi)
            if start != -1:
                # print(rf"{smiles_str[-1]}")
                # print(f"Molecule found. cid: {cid} {smiles_str} (inchi) {inchi}")
                found_molecules[smiles_str] = True
        except Exception as e:
            print(e)
    return found_molecules


if __name__ == "__main__":
    pprint(check_in_pubchem([
        # from pubchem
        "CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C",
        "CC(CN)O",
        "C1=CC=C(C=C1)O",
        "C1=CC=C2C(=C1)C=C(C=N2)NC(=O)C3=C4C=CC=CN4C(=C3)C(=O)C5=CC=C(C=C5)Br",
        "COC1=CC(=CC(=C1OC)OC)C(=O)C[N+]2=CC=CC3=C2C=CC4=CC=CC=C43.[Br-]",

        # not from pubchem
        "CCCCCCCC[Na]CCCC[Na]",
        "ClC1=NC=CC(=N1)NCC(OC)OCOCCCCCC",
	"random stirng", # not it does not work about this, it just reporst that it is not found, because it cannot make a molecule out of this
    ]))
