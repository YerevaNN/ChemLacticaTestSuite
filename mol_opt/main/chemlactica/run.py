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

# sys.path.append("../ChemLactica/ChemLactica/chemlactica")
from chemlactica.mol_opt.optimization import optimize
from chemlactica.mol_opt.utils import MoleculeEntry, generate_random_number

os.environ["TOKENIZERS_PARALLELISM"] = "true"


# def create_optimization_prompts(num_prompts, molecules_bank: MoleculeEntry, max_similars_in_prompt: int, sim_range):
#     prompts = []
#     for i in range(num_prompts):
#         similars_in_prompt = molecules_bank.random_subset(max_similars_in_prompt)
#         prompt = "</s>"
#         for mol in similars_in_prompt:
#             prompt += f"[SIMILAR]{mol.smiles} {generate_random_number(sim_range[0], sim_range[1]):.2f}[/SIMILAR]"
#         prompt += "[START_SMILES]"
#         prompts.append(prompt)
#     return prompts


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


# def create_molecule_entry(output_text):
#     start_smiles_tag, end_smiles_tag = "[START_SMILES]", "[END_SMILES]"
#     start_ind = output_text.find(start_smiles_tag)
#     end_ind = output_text.find(end_smiles_tag)
#     if start_ind == -1 or end_ind == -1:
#         return None

#     generated_smiles = output_text[start_ind+len(start_smiles_tag):end_ind]
#     try:
#         return MoleculeEntry(
#             smiles=generated_smiles,
#             score=None
#         )
#     except:
#         return None


def default_train_condition(num_iter, tol_level, prev_train_iter):
    return num_iter - prev_train_iter >= 3


def tolerance_train_condition(num_iter, tol_level, prev_train_iter):
    return num_iter - prev_train_iter >= 3 and tol_level > 3


def choose_train_condition(name):
    return {
        "default" : default_train_condition,
        "tolerance": tolerance_train_condition
    }[name]


def construct_additional_properties(config):
    
    additional_properties = {
        "qed" : {
            "start_tag": "[QED]",
            "end_tag": "[/QED]",
            "infer_value": lambda entry: f"{generate_random_number(config['qed_range'][0], config['qed_range'][1]):.2f}",
            "calculate_value": lambda entry: f"{qed(entry.mol):.2f}"
        }
    }

    return additional_properties


class ChemLactica_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "chemlactica"

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)

        # additional_properties = construct_additional_properties(config)

        model = AutoModelForCausalLM.from_pretrained(config["checkpoint_path"], torch_dtype=torch.bfloat16).to(config["device"])
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], padding_side="left")
        config["log_dir"] = os.path.join(self.oracle.args.output_dir, 'results_' + self.oracle.task_label + '.log')
        config["rej_sample_config"]["should_train"] = choose_train_condition(config["rej_sample_config"]["train_condition"])
        config["max_possible_oracle_score"] = 1.0
        optimize(
            model=model, tokenizer=tokenizer,
            oracle=self.oracle, config=config
        )