import copy
from utils import logits_utils
import os
from paths import *
from property_ranges import *
import yaml
from dataclasses import asdict

with open('gen_configs.yaml', 'r') as file:
    generation_configs = yaml.full_load(file) 

top_N = 100
n_per_vs_rmse = 4
regexp = "^.*?(?=\\[END_SMILES])"
# torch_dtype = "float32"
torch_dtype = "bfloat16"
device = "cuda:1"
# device = "cuda:0"
# device = 'cpu'
# target_dist = "prior"
target_dist = "uniform"
std_var = 1
gemma_logits_processors_configs = logits_utils.load_processor_config(gemma_logit_config_path)
galactica_logits_processors_configs = logits_utils.load_processor_config(galactica_logit_config_path)
gemma_lp_configs = [asdict(obj) for obj in gemma_logits_processors_configs]
galactica_lp_configs = [asdict(obj) for obj in galactica_logits_processors_configs]


# target_dists = ["prior", "uniform"]
# models = [model_2b_32k_8f45, model_2b_3k_504e]
models = [model_2b_10k_6a86, model_2b_8k_6a86, model_2b_6b_6a86, model_2b_4k_6a86, model_2b_7k_d46c, model_2b_9b_d46c, model_2b_10k_d46c, model_2b_12k_d46c]

evaluation_config = {
    "test_suite":                test_suite,
    "property_range":            property_range,
    "generation_config":         generation_configs["gem_greedy_generation_config"],
    "model_checkpoint_path":     model_2b_18k_0717,
    "tokenizer_path":            gemma_custom_tokenizer_path,
    "logits_processors_configs": None,
    "std_var":                   0,
    "torch_dtype":               torch_dtype,
    "device":                    device,
    "regexp":                    regexp,
    "top_N":                     top_N,
    "target_dist":               target_dist,
    "n_per_vs_rmse":             n_per_vs_rmse,
    "include_eos":               True,
    "include_start_smiles":      False,
    "check_for_novelty":         True,
    "track":                     True,
    "plot":                      True,
    "description":               "CoT_sup_rep01_uniform",
}

evaluation_configs = [evaluation_config]