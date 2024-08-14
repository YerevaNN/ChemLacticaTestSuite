import os
import yaml
import torch
import random
import argparse
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from chemlactica.mol_opt.optimization import optimize
from chemlactica.mol_opt.utils import set_seed

from oracle import SaturnDockingOracle

def set_seed_everywhere(seed: int, device: str):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_default", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    with open(args.config_default, 'r') as f:
        config = yaml.safe_load(f)

    model = AutoModelForCausalLM.from_pretrained(config["checkpoint_path"], torch_dtype=torch.bfloat16).to(config["device"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], padding_side="left")

    seed = config["seed"]
    set_seed_everywhere(seed, config["device"])
    set_seed(seed)

    oracle = SaturnDockingOracle(3000, config["target"], takes_entry=True)

    config["log_dir"] = os.path.join(args.output_dir, f"results_saturn+{config['target']}+seed_{seed}.log")
    config["max_possible_oracle_score"] = oracle.max_possible_oracle_score

    optimize(
        model, tokenizer,
        oracle, config,
        validate_smiles=lambda x: "." not in x,
    )