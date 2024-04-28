from rdkit import Chem
import rdkit
from rdkit.Chem import AllChem
import tempfile
import re
import subprocess
from typing import List, Union
from rdkit.Chem.Descriptors import MolWt
from vina_config import VinaConfig
import random
from transforms import reverse_sigmoid_transformation, double_sigmoid, compute_non_penalty_components
from openbabel import openbabel

# below parameters taken from 
# https://github.com/schwallergroup/augmented_memory/blob/6170fa4181c0bc5b7523e49bdc5bfd2b60f2a6a9/beam_enumeration_reproduce_experiments/drug-discovery-experiments/drd2/docking.json # noqa
NUM_POSES = 1
VINA_EXHAUSTIVENESS = 8
VINA_SEED = 42
MAXIMUM_ITERATIONS = 600
VINA_BINARY_PATH = "/mnt/sxtn2/chem/chemlm_oracles/vina_bin/autodock_vina_1_1_2_linux_x86/bin/vina"


def get_vina_score(smiles_to_score: Union[str, List], vina_params:VinaConfig):
    random.seed(42)
    scores = []

    if isinstance(smiles_to_score, str):
        smiles_to_score = [smiles_to_score]

    for smiles in smiles_to_score:
        mol = smiles_to_reasonable_conformer(smiles)
        ligand_mol_pdbqt = obabel_mol_to_pdbqt_string(mol)
        with tempfile.NamedTemporaryFile(mode="w", delete=True) as temp_file:
            temp_file.write(ligand_mol_pdbqt)
            ligand_mem_path = temp_file.name
            temp_file.seek(0) # move the file cursor back to the start or vina will not read the file and break
            vina_command = f"{VINA_BINARY_PATH}  --center_x {vina_params.centers[0]} --center_y {vina_params.centers[1]} --center_z {vina_params.centers[2]} --size_x {vina_params.box_size[0]} --size_y {vina_params.box_size[1]} --size_z {vina_params.box_size[2]} --ligand {str(ligand_mem_path)} --receptor {vina_params.receptor} --seed {VINA_SEED} --exhaustiveness {VINA_EXHAUSTIVENESS}"
            vina_result = subprocess.run(vina_command, shell=True, capture_output = True)

        pattern = r"\s+\d+\s+(-?\d+\.\d+)\s+\d+\.\d+\s+\d+\.\d+" # don't ask
        weights = [1,1,1]

        # Search for the pattern in the stdout
        stdout = re.findall(pattern, vina_result.stdout.decode())

        vina = float(stdout[0])
        qed = Chem.QED.qed(mol)
        weight = Chem.Descriptors.MolWt(mol)
        print("raw vina:",vina,"raw qed:",qed,"raw weight:",weight)

        vina_result = reverse_sigmoid_transformation(vina_params.transform_sigmoid_low,[vina])[0]
        weight_result = double_sigmoid([10.])
        qed_result = qed # no transformation
        print("transformed vina:",vina_result,"transformed qed:",qed_result,"transformed weight:",weight_result)

        individual_score_values = [[qed_result],[vina_result],[weight_result]]
        total_oracle_result = compute_non_penalty_components(weights,individual_score_values,[smiles])
        print("total score",total_oracle_result)
        scores.append(float(total_oracle_result))
    return scores

def smiles_to_reasonable_conformer(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)

    AllChem.EmbedMolecule(mol,randomSeed=42,useRandomCoords=True)
    AllChem.UFFOptimizeMolecule(mol, maxIters=600)
    mol = Chem.AddHs(mol, addCoords=True)

    return mol


def obabel_mol_to_pdbqt_string(rdmol):
    omol = openbabel.OBMol()
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("pdb", "pdbqt")
    pdb_block = Chem.MolToPDBBlock(rdmol)
    conv.ReadString(omol, pdb_block)
    output_pdbqt_block = conv.WriteString(omol)
    return output_pdbqt_block
    
def main():
    sm_list = [
        'O=C1CC=C(c2ccc(N3CC(c4ccccc4)CC3=O)cc2)CN1',
        'Cc1cc(N2CCCC2)ccc1CNC(=O)c1cc2c(c(C)n1)CCCC2',
        'O=C(NC1C2CC3CC(C2)CC1C3)C1=CN=C(N2CCC3=CC=CC=C3C2)N=N1'
        ]

    vina_params = VinaConfig(centers=[9.93, 5.85, -9.58], box_size=[15, 15, 15],experiment="drd2",transform_sigmoid_low = -16)
    scores = get_vina_score(sm_list, vina_params)
    print("------------")
    print(scores)


if __name__ == "__main__":
    main()
