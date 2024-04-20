from rdkit import Chem
from rdkit.Chem import AllChem
from vina import Vina
from rd_pdbqt import MolToPDBQTBlock
from typing import List, Union
from vina_config import VinaConfig

# below parameters taken from 
# https://github.com/schwallergroup/augmented_memory/blob/6170fa4181c0bc5b7523e49bdc5bfd2b60f2a6a9/beam_enumeration_reproduce_experiments/drug-discovery-experiments/drd2/docking.json # noqa
NUM_POSES = 1
VINA_SEED = 42
MAXIMUM_ITERATIONS = 600


def get_vina_score(smiles_to_score: Union[str, List], vina_params:VinaConfig):
    scores = []
    vina_obj = Vina(sf_name="vina", seed=vina_params.seed, verbosity=0)

    if isinstance(smiles_to_score, str):
        smiles_to_score = [smiles_to_score]
    vina_obj.set_receptor(vina_params.receptor)
    vina_obj.compute_vina_maps(center=vina_params.centers, box_size=vina_params.box_size)

    for smiles in smiles_to_score:
        try:
            ligand_mol_pdbqt = smiles_to_reasonable_conformer_pdbqt(smiles)
            vina_obj.set_ligand_from_string(ligand_mol_pdbqt)
            energies = vina_obj.dock(
                exhaustiveness=vina_params.exhaustiveness, n_poses=vina_params.num_poses
            )
            energies = vina_obj.optimize()
            result = energies[0]  # get lowest energy binding mode
        except Exception as e:
            result = None
            print(e)
        scores.append(result)
    return scores


def smiles_to_reasonable_conformer_pdbqt(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol, maxIters=MAXIMUM_ITERATIONS)
    pdbqt_string = MolToPDBQTBlock(mol)
    return pdbqt_string


def main():
    smiles = "CCO"
    vina_params = VinaConfig(centers=[9.93, 5.85, -9.58], box_size=[15, 15, 15],experiment="drd2")
    score = get_vina_score(smiles, vina_params)
    print(score)


if __name__ == "__main__":
    main()
