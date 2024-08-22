import os
import yaml
import torch
import random
import argparse
import numpy as np
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from chemlactica.mol_opt.optimization import optimize
from chemlactica.mol_opt.utils import set_seed

from oracle import SaturnDockingOracle, IllustrativeExperimentOracle

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
    parser.add_argument("--illustrative", type=bool, default=False)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target", type=Optional[str], default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    args = parse_arguments()
    is_illustrative = args.illustrative

    budget = 1000 if is_illustrative else 3000
    smiles_validator = (lambda x: True) if is_illustrative else (lambda x: "." not in x)
    oracle_class = IllustrativeExperimentOracle if is_illustrative else SaturnDockingOracle

    with open(args.config_default, 'r') as f:
        config = yaml.safe_load(f)
        
    oracle_kwargs = {"target": args.target} if not is_illustrative else {}
    run_name_prefix = "illustrative" if is_illustrative else f"saturn+{args.target}"

    model = AutoModelForCausalLM.from_pretrained(config["checkpoint_path"], torch_dtype=torch.bfloat16).to(config["device"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], padding_side="left")

    seed = args.seed
    set_seed_everywhere(seed, config["device"])
    set_seed(seed)

    oracle = oracle_class(budget, takes_entry=True, **oracle_kwargs)

    config["log_dir"] = os.path.join(args.output_dir, f"results_{run_name_prefix}+seed_{seed}.log")
    config["max_possible_oracle_score"] = oracle.max_possible_oracle_score

    optimize(
        model, tokenizer,
        oracle, config,
        validate_smiles=smiles_validator,
    )