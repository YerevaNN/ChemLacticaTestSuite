import os
import yaml
import json
import argparse
from datetime import datetime
from typing import List

import torch

from paths import model_paths, tokenizer_paths
from property_ranges import *
from property_prediction import run_property_predictions
from conditional_generation import Property2Mol

with open('gen_configs.yaml', 'r') as file:
    generation_configs = yaml.full_load(file) 

def parse_args():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model or a known path name"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the tokenizer or a known path name"
    )
    parser.add_argument(
        "--properties_name",
        type=str,
        required=True,
        default="all",
        help="List of property names to evaluate"
    )
    parser.add_argument(
        "--generation_config",
        type=str,
        default="default_config.json",
        help="Generation configuration dict name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cuda/cpu)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for model computation"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="../results",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--target_dist",
        type=str,
        default="prior",
        choices=["prior", "uniform"],
        help="Target distribution for conditional generation samples"
    )
    parser.add_argument(
        "--pdf_export",
        action="store_true",
        help="Enable PDF export of results"
    )
    parser.add_argument(
        "--no_CoT",
        action="store_true",
        help="Disable Chain-of-Thought reasoning"
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Include detailed descriptions in output"
    )
    parser.add_argument(
        "--no_plot",
        action="store_false",
        help="Generate plots for analysis results"
    )
    parser.add_argument(
        "--no_tracking",
        action="store_true",
        help="Enable experiment tracking"
    )
    parser.add_argument(
        "--partial_property_range",
        action="store_true",
        help="Evaluate full range of property values"
    )

    args = parser.parse_args()
    return args

def start_log_file(model_path, description, result_path, generation_config):
    model_name = model_path.split("/")[-4]
    model_hash = model_path.split("/")[-3][:4]
    model_checkpoint = model_path.split("/")[-2][11:]
    results_path = os.path.join( os.path.join(result_path, "property_2_Mol/"), \
                                        f"{datetime.now().strftime('%Y-%m-%d-%H:%M')}"\
                                        f"-{model_name}-{model_hash}-{model_checkpoint}"\
                                        f"-{generation_config['name']}")
    if description:
        description += "/"
        results_path += "-" + description
    else:
        results_path += "/"
    print(f'results_path = {results_path}\n')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    log_file = open(results_path + 'full_log.txt', 'w+')
    
    log_file.write(f'results of property to molecule test performed at '\
                        f'{datetime.now().strftime("&Y-%m-%d, %H:%M")}\n')
    log_file.write(f'evaluation config: \n{json.dumps(generation_config, indent=4)}\n')

    return results_path, log_file

def main():
    args = parse_args()
    model_path = model_paths.get(args.model_path, args.model_path)
    tokenizer_path = tokenizer_paths.get(args.tokenizer_path, args.tokenizer_path)
    generation_config = generation_configs[args.generation_config]
    results_path, log_file = start_log_file(model_path, args.description,  args.results_path, generation_config)
    pp_results, sim_pp_results, sim_cg_results = run_property_predictions(model_path, 
                                                                          tokenizer_path,
                                                                          args.device, 
                                                                          args.properties_name, 
                                                                          generation_config,
                                                                          results_path,
                                                                          log_file,
                                                                          pdf_export=False)
    
    evaluation_config = {
        "test_suite":                test_suite,
        "property_range":            property_range,
        "generation_config":         generation_config,
        "model_checkpoint_path":     model_path,
        "tokenizer_path":            tokenizer_path,
        "logits_processors_configs": None,
        "std_var":                   0,
        "torch_dtype":               args.dtype,
        "device":                    args.device,
        "target_dist":               args.target_dist,
        "include_start_smiles":      args.no_CoT,
        "check_for_novelty":         True,
        "track":                     args.no_tracking,
        "plot":                      args.no_plot,
        "description":               args.description,
        "log_file":                  log_file,
        "result_path":               results_path,
    }
    conditional_generation = Property2Mol(**evaluation_config)
    cg_results = conditional_generation.run_property_2_Mol_test()

    results_file = open(results_path + 'final_results.txt', 'w+')
    results_file.write(f"property prediction results: \n{json.dumps(pp_results, indent=4)}\n")
    results_file.write(f"similarity prediction results: \n{json.dumps(sim_pp_results, indent=4)}\n")
    results_file.write(f"similarity conditional generation results: \n{json.dumps(sim_cg_results, indent=4)}\n")
    results_file.write(f"conditional generation results: \n{json.dumps(cg_results, indent=4)}\n")
    log_file.close()
    results_file.close()

if __name__ == "__main__":
    main()
