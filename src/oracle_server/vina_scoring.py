from rdkit import Chem
from multiprocessing import Pool
from functools import partial
import time
import traceback
import argparse
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

def calculate_vina_comb_score(smiles,num_cpu,vina_binary_path,vina_params:VinaConfig):
    #print(smiles)
    try:
        mol = smiles_to_reasonable_conformer(smiles)
        ligand_mol_pdbqt = obabel_mol_to_pdbqt_string(mol)
        with tempfile.NamedTemporaryFile(mode="w", delete=True) as temp_file:
            temp_file.write(ligand_mol_pdbqt)
            ligand_mem_path = temp_file.name
            temp_file.seek(0) # move the file cursor back to the start or vina will not read the file and break
            if num_cpu is not None:
                cpu_command_component = f"--cpu {num_cpu}"
            else: 
                cpu_command_component = ""
            vina_command = f"{vina_binary_path} --center_x {vina_params.centers[0]} --center_y {vina_params.centers[1]} --center_z {vina_params.centers[2]} --size_x {vina_params.box_size[0]} --size_y {vina_params.box_size[1]} --size_z {vina_params.box_size[2]} --ligand {str(ligand_mem_path)} --receptor {vina_params.receptor} --seed {VINA_SEED} --exhaustiveness {VINA_EXHAUSTIVENESS} {cpu_command_component}"
            vina_result = subprocess.run(vina_command, shell=True, capture_output = True)

        pattern = r"\s+\d+\s+(-?\d+\.\d+)\s+\d+\.\d+\s+\d+\.\d+" # don't ask
        weights = [1,1,1]

        # Search for the pattern in the stdout
        stdout = re.findall(pattern, vina_result.stdout.decode())
        # print("888888888888888")
        # print(stdout)
        # print("888888888888888")

        vina = float(stdout[0])
        qed = Chem.QED.qed(mol)
        weight = Chem.Descriptors.MolWt(mol)
        #print("raw vina:",vina,"raw qed:",qed,"raw weight:",weight)

        vina_result = reverse_sigmoid_transformation(vina_params.transform_sigmoid_low,[vina])[0]
        weight_result = double_sigmoid([10.])
        qed_result = qed # no transformation
        #print("transformed vina:",vina_result,"transformed qed:",qed_result,"transformed weight:",weight_result)

        individual_score_values = [[qed_result],[vina_result],[weight_result]]
        total_oracle_result = compute_non_penalty_components(weights,individual_score_values,[smiles])
        #print("total score",total_oracle_result)
        if isinstance(total_oracle_result, list):
            #print("list of length:",len(total_oracle_result))
            total_oracle_result = total_oracle_result[0]

        score = float(total_oracle_result)
    except Exception as e:
        print("we excepted the following:",e)
        print(traceback.format_exc())
        score = float(0)

    return score



def get_vina_score(smiles_to_score: Union[str, List], vina_binary_path, vina_params:VinaConfig, num_cpu:int):
    random.seed(42)
    scores = []
    print("how many smiles to score",len(smiles_to_score))

    if isinstance(smiles_to_score, str):
        smiles_to_score = [smiles_to_score]
    #new_func = partial(process_item, arg1, arg2, arg3)
    print("input_data",smiles_to_score)
    curr_vina_function = partial(calculate_vina_comb_score,num_cpu = num_cpu,vina_binary_path = vina_binary_path,vina_params = vina_params)
    with Pool(1) as p:
        scores = p.map(curr_vina_function, smiles_to_score)


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
    start = time.time()
    sm_list = [
        'O=C1CC=C(c2ccc(N3CC(c4ccccc4)CC3=O)cc2)CN1',
        'Cc1cc(N2CCCC2)ccc1CNC(=O)c1cc2c(c(C)n1)CCCC2',
        'O=C(NC1C2CC3CC(C2)CC1C3)C1=CN=C(N2CCC3=CC=CC=C3C2)N=N1'
        ]

    vina_params = VinaConfig(centers=[9.93, 5.85, -9.58], box_size=[15, 15, 15],experiment="drd2",transform_sigmoid_low = -16)
    scores = get_vina_score(sm_list, vina_params)
    scores = get_vina_score(sm_list,vina_binary_path = "/nfs/ap/mnt/sxtn2/chem/chemlm_oracles/vina_bin/autodock_vina_1_1_2_linux_x86/bin/vina", vina_params = vina_params,num_cpu=8)
    print("------------")
    print(scores)
    print("time elapsed",time.time() - start)


if __name__ == "__main__":
    #args = parser.parse_args()
    main()
