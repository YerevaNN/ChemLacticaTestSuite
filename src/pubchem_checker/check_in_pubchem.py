from tqdm import tqdm
from stringzilla import Str, Strs, File
from rdkit import Chem
from rdkit import RDLogger
from typing import List, Dict
from pprint import pprint

RDLogger.DisableLog('rdApp.*')

sorted_inchi_str = Str(File('/nfs/dgx/raid/chem/pubchem_inchi/sorted_inchi_mols'))
sorted_inchi_molecules = sorted_inchi_str.split(separator='\n')


def binary_search(inchi_string):
    l = 0
    r = len(sorted_inchi_molecules) - 1;
    while l <= r:
        mid = (l + r) // 2;
        if sorted_inchi_molecules[mid] == inchi_string:
            return mid
        if sorted_inchi_molecules[mid] > inchi_string:
            r = mid - 1
        else:
            l = mid + 1
    return -1


def check_in_pubchem(molecule_list: List[str]):
    found_molecules: Dict[str, int] = {smiles: False for smiles in molecule_list}
    for smiles_str in tqdm(molecule_list):
        try:
            inchi = Chem.MolToInchi(Chem.MolFromSmiles(smiles_str))
            start = binary_search(inchi)
            # sorted_inchi_str.find(inchi)
            if start != -1:
                found_molecules[smiles_str] = True
        except KeyboardInterrupt as e:
            print(e)
            return
        except Exception as e:
            print(e)
    return found_molecules


if __name__ == "__main__":
    with open("generated_molecules.txt", "r") as _f:
        molecules = [l.rstrip("\n") for l in _f.readlines()]
    # pprint(check_in_pubchem([
    #     # from pubchem
    #     "CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C",
    #     "CC(CN)O",
    #     "C1=CC=C(C=C1)O",
    #     "C1=CC=C2C(=C1)C=C(C=N2)NC(=O)C3=C4C=CC=CN4C(=C3)C(=O)C5=CC=C(C=C5)Br",
    #     "COC1=CC(=CC(=C1OC)OC)C(=O)C[N+]2=CC=CC3=C2C=CC4=CC=CC=C43.[Br-]",

    #     # not from pubchem
    #     "CCCCCCCC[Na]CCCC[Na]",
    #     "ClC1=NC=CC(=N1)NCC(OC)OCOCCCCCC",
	#     "random stirng", # not it does not work about this, it just reporst that it is not found, because it cannot make a molecule out of this
    # ]))
    check_in_pubchem(molecules)
    # # print(inchi_mols[0])
