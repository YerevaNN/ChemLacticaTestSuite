from oracles.vina_oracle import VinaOracle
from oracles.oracle_utils import vina_prompts_post_processor
from chemlactica.mol_opt.utils import MoleculeEntry

import pandas as pd
from transformers import OPTForCausalLM, AutoTokenizer
import torch
import numpy as np
import logging
import datetime
import argparse
import gc
import time
import os
import sys
import yaml
import random
from typing import List
from tqdm import tqdm
#from sas_sim_oracle import InheritedLeadOptimizationSASOracle
from rdkit import Chem
import multiprocessing
from functools import partial

from chemlactica.mol_opt.optimization import optimize, create_molecule_entry
from chemlactica.mol_opt.utils import set_seed, generate_random_number, MoleculeEntry

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--input_mols_csv_path", type=str, required=False)
    parser.add_argument("--config_default", type=str, required=False, default="hparams_default.yaml")
    parser.add_argument("--n_runs", type=int, required=False, default=1)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    config = yaml.safe_load(open(args.config_default))
    print(config)
    
    model = OPTForCausalLM.from_pretrained(config["checkpoint_path"], torch_dtype=torch.bfloat16).to(config["device"])
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], padding_side="left")

    
    formatted_date_time = datetime.datetime.now().strftime("%Y-%m-%d")
    base = f"results/{formatted_date_time}"
    os.makedirs(base, exist_ok=True)
    v = 0
    name = args.run_name + "+".join(config["strategy"])
    while os.path.exists(os.path.join(base, f"{name}-{v}")):
        v += 1
    output_dir = os.path.join(base, f"{name}-{v}")
    os.makedirs(output_dir, exist_ok=True)



    if args.input_mols_csv_path is not None:
        molecules_df = pd.read_csv(args.input_mols_csv_path)
        lead_molecules = molecules_df.smiles.to_list()
    else:
        # if we don't start with molecules to optimize, we generate a random SMILES with no conditioning
        inputs = "</s>"
        inputs = tokenizer(input, return_tensors="pt").to(config["device"])
        output = model.generate(inputs = input, **config["generation_config"])
        lead_molecule_output_text = tokenizer.decode(output[i], skip_special_tokens=False) for i in range(len(output))
        lead_molecules = [create_molecule_entry(lead_molecule_output_text)]


    seeds = [2, 9, 31]
    acc_success_rate = 0
    acc_max_sim_sum = 0
    for seed in seeds[:args.n_runs]:
        set_seed(seed)
        max_sim_sum = 0
        num_optimized = 0

        oracle_url = 'http://172.26.26.101:5006/oracles/vina/drd2'
        oracle = VinaOracle(url = oracle_url, max_oracle_calls=5000)

        for i in tqdm(range(len(lead_molecules))):
            oracle.reset()
            lead = lead_molecules[i]
            if isinstance(lead, str):
                lead_mol = MoleculeEntry(lead)

            oracle.set_lead_entry(lead_mol)
            config["prompts_post_processor"] = vina_prompts_post_processor
            print(f"Optimizing {lead}")
            config["log_dir"] = os.path.join(output_dir, f'mol-{i}-{seed}.log')
            optimize(model, tokenizer, oracle, config)

        print(oracle._calculate_metrics())
    #     results_dict = oracle._calculate_metrics()
    #     acc_success_rate += results_dict["success_rate"]
    #     acc_max_sim_sum += results_dict["max_sim_sum"]
    #     max_sim_mean = results_dict["max_sim_mean"]
    #     print(f"Sum of similarities {max_sim_mean:.4f}, seed {seed}")
    #     print(f"Success rate {results_dict['success_rate']:.4f}, seed {seed}")

    # print(f"Sum of similarities across {args.n_runs} runs {acc_max_sim_sum / args.n_runs:.4f}")
    # print(f"Success rate across {args.n_runs} runs {acc_success_rate / args.n_runs:.4f}")


# def main():
#     oracle_url =  'http://ap.yerevann.com:5006/oracles/vina/drd2'
#     my_mol = MoleculeEntry(smiles="CC(C)C1=CC=CC=C1C(=O)O")
#     oracle = VinaOracle(url = oracle_url, max_oracle_calls=100)
#     result = oracle(my_mol)
#     print("my score", result)

if __name__ == "__main__":
    main()
