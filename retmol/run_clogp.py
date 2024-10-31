import pandas as pd
from transformers import OPTForCausalLM, AutoTokenizer
import torch
import numpy as np
import logging
import yaml
import datetime
import argparse
import os
from typing import List
from tqdm import tqdm
from utils import LeadOptimizationPlogPOracle
from rdkit.Chem.QED import qed
from chemlactica.mol_opt.utils import generate_random_number, set_seed, MoleculeEntry, tanimoto_dist_func
from chemlactica.mol_opt.optimization import optimize

# RetMol/data/zinc/inference_inputs/opt.test.logP-SA

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--clogp_csv_path", type=str, required=True)
    parser.add_argument("--config_default", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--n_runs", type=int, required=False, default=1)
    args = parser.parse_args()
    return args


def construct_additional_properties(lead_mol, config):
    
    def get_infer_value(entry):
        return f"{lead_mol.smiles} {generate_random_number(config['sim_range'][0], config['sim_range'][1]):.2f}"
    
    def get_train_value(entry):
        return f"{lead_mol.smiles} {tanimoto_dist_func(lead_mol.fingerprint, entry.fingerprint):.2f}"

    additional_properties = {
        # "qed" : {
        #     "start_tag": "[QED]",
        #     "end_tag": "[/QED]",
        #     "infer_value": lambda entry: f"{generate_random_number(config['qed_range'][0], config['qed_range'][1]):.2f}",
        #     "calculate_value": calculate_qed
        # }
        "similar": {
            "start_tag": "[SIMILAR]",
            "end_tag": "[/SIMILAR]",
            "infer_value": get_infer_value,
            "calculate_value": get_train_value
        }
    }

    return additional_properties


if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.safe_load(open(args.config_default))
    print(config)

    model = OPTForCausalLM.from_pretrained(config["checkpoint_path"], torch_dtype=torch.bfloat16).to(config["device"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], padding_side="right")

    molecules_df = pd.read_csv(args.clogp_csv_path, names=["smiles", "clogp"], header=None, sep=" ")
    lead_molecules = molecules_df.smiles.to_list()

    seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    acc_success_rate = 0
    for seed in seeds[:args.n_runs]:
        set_seed(seed)
        total = 0
        num_optimized = 0
        for i in tqdm(range(len(lead_molecules))):
            lead = lead_molecules[i]
            lead_mol = MoleculeEntry(lead)

            additional_properties = construct_additional_properties(lead_mol, config)
            # additional_properties = {}

            print(f"Optimizing {lead}")
            oracle = LeadOptimizationPlogPOracle(lead_mol, max_oracle_calls=10000, sim_bound=0.4)
            config["log_dir"] = os.path.join(args.output_dir, f'mol-{i}-{seed}.log')
            # config["rej_sample_config"]["should_train"] = choose_train_condition(config["rej_sample_config"]["train_condition"])
            config["max_possible_oracle_score"] = 2.0
            optimize(
                model, tokenizer,
                oracle, config,
                additional_properties
            )
            num_optimized += oracle.found_opt_mol
            total += 1
            print(f"Current success rate {num_optimized / total:.4f}")
        success_rate = num_optimized / len(lead_molecules)
        acc_success_rate += success_rate
        print(f"Success rate {success_rate:.4f}, seed {seed}")
    print(f"Success rate across {args.n_runs} runs {acc_success_rate / args.n_runs:.4f}")
