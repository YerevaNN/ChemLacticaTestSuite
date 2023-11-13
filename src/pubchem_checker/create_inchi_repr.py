from tqdm import tqdm

from rdkit import Chem
from stringzilla import Str, Strs, File
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


if __name__ == "__main__":
    text = Str(File('CID-SMILES'))
    text = text.split(separator="\n")
    with open("inchi", "w") as _f:
        for i in tqdm(range(len(text))):
            cid, smiles = text[i].split("\t")
            smiles_str = "".join([c for c in smiles])

            try:
                inchi_str = Chem.MolToInchi(Chem.MolFromSmiles(smiles_str))
                _f.write(f"{cid}\t{inchi_str}\n")
            except:
                pass


def sort_inchi_mols():
    inchi_mols = Str(File('sorted_inchi')).split(separator='\n')
    inchi_mols.sort()
    with open("sorted_inchi_mols", "w") as _f:
        for molecule in inchi_mols:
            # print(molecule)
            _f.write("".join(molecule) + "\n")