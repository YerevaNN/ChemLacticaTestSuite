import os
import yaml
import torch
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from chemlactica.mol_opt.optimization import optimize
from chemlactica.mol_opt.utils import set_seed

# from saturn.utils.utils import set_seed_everywhere
from oracle import SaturnDockingOracle

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_default", type=str, required=True)
    parser.add_argument("--n_runs", type=int, required=False, default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    with open(args.config_default, 'r') as f:
        config = yaml.safe_load(f)

    model = AutoModelForCausalLM.from_pretrained(config["checkpoint_path"], torch_dtype=torch.bfloat16).to(config["device"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], padding_side="left")

    seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for i in range(args.n_runs):
        # set_seed_everywhere(seeds[i])
        set_seed(seeds[i])

        oracle = SaturnDockingOracle(1000, config["target"], takes_entry=True)

        config["log_dir"] = os.path.join(args.output_dir, f"results_saturn+weight+num_run_{i}.log")
        config["max_possible_oracle_score"] = oracle.max_possible_oracle_score

        optimize(
            model, tokenizer,
            oracle, config
        )