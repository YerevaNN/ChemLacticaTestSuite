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
from oracles.sas_sim_oracle import InheritedLeadOptimizationSASOracle
from rdkit import Chem
import multiprocessing
from functools import partial

from chemlactica.mol_opt.optimization import optimize
from chemlactica.mol_opt.utils import set_seed, generate_random_number, MoleculeEntry

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--supnatural_csv_path", type=str, required=True)
    parser.add_argument("--config_default", type=str, required=False, default="hparams_default.yaml")
    parser.add_argument("--n_runs", type=int, required=False, default=1)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    config = yaml.safe_load(open(args.config_default))
    print(config)
    
    model = OPTForCausalLM.from_pretrained(config["checkpoint_path"], torch_dtype=torch.bfloat16).to(config["device"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], padding_side="left")

    molecules_df = pd.read_csv(args.supnatural_csv_path)
    lead_molecules = molecules_df.smiles.to_list()

    formatted_date_time = datetime.datetime.now().strftime("%Y-%m-%d")
    base = f"results/{formatted_date_time}"
    os.makedirs(base, exist_ok=True)
    v = 0
    name = "+".join(config["strategy"]) + args.run_name
    while os.path.exists(os.path.join(base, f"{name}-{v}")):
        v += 1
    output_dir = os.path.join(base, f"{name}-{v}")
    os.makedirs(output_dir, exist_ok=True)

    seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    acc_success_rate = 0
    acc_max_sim_sum = 0
    for seed in seeds[:args.n_runs]:
        set_seed(seed)
        max_sim_sum = 0
        num_optimized = 0
        oracle = InheritedLeadOptimizationSASOracle(3.0, max_oracle_calls=1000)
        for i in tqdm(range(len(lead_molecules))):
            oracle.reset()

            lead = lead_molecules[i]
            lead_mol = MoleculeEntry(lead)
            oracle.set_lead_entry(lead_mol)
            def prompts_post_processor(prompt):
                prompt += f"[SIMILAR]{lead_mol.smiles} {generate_random_number(config['sim_range'][0], config['sim_range'][0])}[/SIMILAR]"
                prompt += f"[SAS]{generate_random_number(config['sas_range'][0], config['sas_range'][1]):.2f}[/SAS]"
                return prompt
            config["prompts_post_processor"] = prompts_post_processor

            print(f"Optimizing {lead}")
            config["log_dir"] = os.path.join(output_dir, f'mol-{i}-{seed}.log')
            optimize(model, tokenizer, oracle, config)

        print(oracle._calculate_metrics())
        results_dict = oracle._calculate_metrics()
        acc_success_rate += results_dict["success_rate"]
        acc_max_sim_sum += results_dict["max_sim_sum"]
        max_sim_mean = results_dict["max_sim_mean"]
        print(f"Sum of similarities {max_sim_mean:.4f}, seed {seed}")
        print(f"Success rate {results_dict['success_rate']:.4f}, seed {seed}")

    print(f"Sum of similarities across {args.n_runs} runs {acc_max_sim_sum / args.n_runs:.4f}")
    print(f"Success rate across {args.n_runs} runs {acc_success_rate / args.n_runs:.4f}")


if __name__ == "__main__":
    main()
