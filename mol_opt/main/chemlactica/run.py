import tdc
from main.optimizer import BaseOptimizer
from transformers import AutoModelForCausalLM, AutoTokenizer
# from main.chemlactica.utils import PMOMoleculeEntry, PMOMoleculeBank, generate_random_number
import gc
import torch
import multiprocessing
import os
import time
import sys
from copy import deepcopy
import numpy as np
from rdkit.Chem.QED import qed
from rdkit.Chem import rdMolDescriptors, Descriptors

# sys.path.append("../ChemLactica/ChemLactica/chemlactica")
from chemlactica.mol_opt.optimization import optimize
from chemlactica.mol_opt.utils import MoleculeEntry, generate_random_number, tanimoto_dist_func, canonicalize

os.environ["TOKENIZERS_PARALLELISM"] = "true"


api_evaluated_oracle_names = ["gsk3b", "jnk3", "drd2"]

class APIOracle(tdc.Oracle):
    
    def __init__(self, name):
        print(name + " is an API based oracle")
        self.name = name
        self.evaluator_func = None
        self.assign_evaluator()

    def assign_evaluator(self):
        import requests
        def evaluator(smiles):
            url = "http://172.26.26.16:4300"
            response = requests.post(url + "/" + self.name, json=smiles)
            # Check the response
            if response.status_code == 200:
                return response.json()[f"{self.name}_score"]
            else:
                print("Error:", response.text)
            return 0.0
        self.evaluator_func = evaluator

    def __call__(self, *args, **kwargs):
        return self.evaluator_func(*args, **kwargs)


def construct_median1_additional_prop(config):

    camphor_molecule = MoleculeEntry("CC1(C)C2CCC1(C)C(=O)C2")
    menthol_molecule = MoleculeEntry("CC(C)C1CCC(C)CC1O")

    p = 1 - tanimoto_dist_func(camphor_molecule.fingerprint, menthol_molecule.fingerprint)
    print("median1, the value of p is ", p)
    
    additional_properties = {}
    additional_properties["sim_camphor"] = {
        "start_tag": "[SIMILAR]",
        "end_tag": "[/SIMILAR]",
        "infer_value": lambda entry: f"{camphor_molecule.smiles} {1 - p / 2:.2f}",
        "calculate_value": lambda entry: f"{camphor_molecule.smiles} {tanimoto_dist_func(camphor_molecule.fingerprint, entry.fingerprint):.2f}"
    }

    additional_properties["sim_menthol"] = {
        "start_tag": "[SIMILAR]",
        "end_tag": "[/SIMILAR]",
        "infer_value": lambda entry: f"{menthol_molecule.smiles} {1 - p / 2:.2f}",
        "calculate_value": lambda entry: f"{menthol_molecule.smiles} {tanimoto_dist_func(menthol_molecule.fingerprint, entry.fingerprint):.2f}"
    }

    return additional_properties


def construct_sitagliptin_mpo_additional_prop(config):

    sitagliptin_molecule = MoleculeEntry("Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F")
    
    additional_properties = {}
    additional_properties["sim_sitagliptin"] = {
        "start_tag": "[SIMILAR]",
        "end_tag": "[/SIMILAR]",
        "infer_value": lambda entry: f"{sitagliptin_molecule.smiles} 0.99",
        "calculate_value": lambda entry: f"{sitagliptin_molecule.smiles} {tanimoto_dist_func(sitagliptin_molecule.fingerprint, entry.fingerprint):.2f}"
    }
    additional_properties["clogp"] = {
        "start_tag": "[CLOGP]",
        "end_tag": "[/CLOGP]",
        "infer_value": lambda entry: f"2.02",
        "calculate_value": lambda entry: f"{Descriptors.MolLogP(entry.mol):.2f}"
    }
    additional_properties["tpsa"] = {
        "start_tag": "[TPSA]",
        "end_tag": "[/TPSA]",
        "infer_value": lambda entry: f"77.04",
        "calculate_value": lambda entry: f"{rdMolDescriptors.CalcTPSA(entry.mol):.2f}"
    }
    # additional_properties["formula"] = {
    #     "start_tag": "[FORMULA]",
    #     "end_tag": "[/FORMULA]",
    #     "infer_value": lambda entry: f"C16H15F6N5O",
    #     "calculate_value": lambda entry: rdMolDescriptors.CalcMolFormula(entry.mol)
    # }

    return additional_properties


def construct_scaffold_hop_additional_prop(config):

    pharmacophor_molecule = MoleculeEntry("CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C")
    
    additional_properties = {}
    additional_properties["sim_pharmacophor"] = {
        "start_tag": "[SIMILAR]",
        "end_tag": "[/SIMILAR]",
        "infer_value": lambda entry: f"{pharmacophor_molecule.smiles} 0.80",
        "calculate_value": lambda entry: f"{pharmacophor_molecule.smiles} {tanimoto_dist_func(pharmacophor_molecule.fingerprint, entry.fingerprint):.2f}"
    }

    return additional_properties


class ChemLactica_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "chemlactica"

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)

        # additional_properties = construct_sitagliptin_mpo_additional_prop(config)
        # additional_properties = construct_median1_additional_prop(config)

        model = AutoModelForCausalLM.from_pretrained(config["checkpoint_path"], torch_dtype=torch.float32).to(config["device"])
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], padding_side="left")
        config["log_dir"] = os.path.join(self.oracle.args.output_dir, 'results_' + self.oracle.task_label + '.log')
        config["max_possible_oracle_score"] = 1.0
        optimize(
            model=model, tokenizer=tokenizer,
            oracle=self.oracle, config=config,
            # additional_properties=additional_properties
        )