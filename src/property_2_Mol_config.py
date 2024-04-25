import copy

test_suite = {
    "sas": {
        "input_properties": ["sas"],
        "target_properties": ["sas"]
    },
    "qed": {
        "input_properties": ["qed"],
        "target_properties": ["qed"]
    },
    "weight": {
        "input_properties": ["weight"],
        "target_properties": ["weight"]
    },
    "clogp": {
        "input_properties": ["clogp"],
        "target_properties": ["clogp"]
    },
    "tpsa": {
        "input_properties": ["tpsa"],
        "target_properties": ["tpsa"]
    },
    # "similarity": {
    #     "input_properties": ["similarity"],
    #     "target_properties": ["similarity"]
    # }
}

mock_test_suite = {
    "clogp": {
        "input_properties": ["clogp"],
        "target_properties": ["clogp"]
    }
}

property_range = {
    "sas": {
        "range": (1.1, 10),
        "step":  0.1,
        "mean": 2.94,
        "std": 0.82,
        "smiles": ""
    },
    "qed": {
        "range": (0.01, 1),
        "step":  0.01,
        "mean": 0.60,
        "std": 0.22,
        "smiles": ""
    },
    "similarity": {
        "range": (0.01, 1),
        "step":  0.01,
        "smiles": " C1=CC=C(C=C1)C2=NC(=CC3=CC=C(C=C3)[N+](=O)[O-])C(=O)O2",
        "mean": 0.734 # TODO need to update
    },
    "weight": {
        "range": (100.1, 1000.1),
        "step":  1,
        "mean": 366,
        "std": 166,
        "smiles": ""
    },
    "clogp": {
        "range": (1.1, 10),
        "step":  0.1,
        "mean": 3.49,
        "std": 1.53,
        "smiles": ""
    },
    "tpsa": {
        "range": (0, 100.5),
        "step":  0.5,
        "mean": 57.42,
        "std": 23.24,
        "smiles": ""
    },
}  

mock_property_range = {
    "clogp": {
        "range": (3.2, 10),
        "step":  1,
        "mean": 3,
        "smiles": ""
    },
    "sas": {
        "range": (1.2, 10),
        "step":  1,
        "mean": 2.8,
        "smiles": ""
    },
    "qed": {
        "range": (0.01, 1),
        "step":  .1,
        "mean": 0.75,
        "smiles": ""
    },
    "weight": {
        "range": (250, 350),
        "step":  .1,
        "mean": 290,
        "smiles": ""
    },
    "tpsa": {
        "range": (50.1, 70),
        "step":  0.1,
        "mean": 57.42,
        "std": 23.24,
        "smiles": ""
    },
    # "similarity":
    # {
    #     "range": (0.01, 1),
    #     "step":  0.01,
    #     "smiles": " CC(CN)O",
    #     "mean": 1000 # TODO need to update
    # },
}

greedy_generation_config = {
    "name": "greedy",
    "multiple_rounds_generation": False,
    "config": {
        "eos_token_id": 20,
        "max_new_tokens": 300,
        "do_sample": False,  
        "num_return_sequences": 1,
        "num_beams": 1,
        "return_dict_in_generate":True,
        "output_scores":True
    }
}

greedy_beam_generation_config = {
    "name": "greedy_beam=12",
    "multiple_rounds_generation": False,
    "config": {
        "eos_token_id": 20,
        "max_new_tokens": 300,
        "length_penalty": -7,
        "repetition_penalty": 1.0, 
        "diversity_penalty": 1.0,
        "num_beam_groups": 6,
        "do_sample": False,  
        "num_return_sequences": 12,
        "num_beams": 12,
        "return_dict_in_generate": True,
        "output_scores": True,
        "renormalize_logits": True
    }
}

greedy_beam6_generation_config = {
    "name": "greedy_beam=6",
    "multiple_rounds_generation": False,
    "config": {
        "eos_token_id": 20,
        "max_new_tokens": 300,
        "length_penalty": -7,
        "repetition_penalty": 1.0, 
        "diversity_penalty": 1.0,
        "num_beam_groups": 3,
        "do_sample": False,  
        "num_return_sequences": 6,
        "num_beams": 6,
        "return_dict_in_generate": True,
        "output_scores": True,
        "renormalize_logits": True
    }
}
# greedy_beam_generation_config["name"] = f'{greedy_beam_generation_config["config"]["num_beams"]=},'\
#     f'{greedy_beam_generation_config["config"]["length_penalty"]=},{greedy_beam_generation_config["config"]["repetition_penalty"]=},'\
#     f'{greedy_beam_generation_config["config"]["diversity_penalty"]=},{greedy_beam_generation_config["config"]["num_beam_groups"]=},'\
#     f'{greedy_beam_generation_config["config"]["num_return_sequences"]=},{greedy_beam_generation_config["config"]["max_new_tokens"]=},'

nongreedy_generation_config = {
    "name": "nongreedy_5of20",
    "top_N": 5,
    "multiple_rounds_generation": False,
    "target_dist": "prior",
    "config": {
        "eos_token_id": 20,
        "max_new_tokens": 300,
        "top_k": None,
        "top_p": 1.0,
        "do_sample": True,  
        "num_return_sequences": 20,
        "num_beams": 1,
        "return_dict_in_generate":True,
        "output_scores":True
    }
}

nongreedy_calibration_generation_config = {
    "name": "nongreedy_calibration_1k",
    "top_N": 1000,
    "total_gen_range": 10,
    "multiple_rounds_generation": True,
    "config": {
        "eos_token_id": 20,
        "max_new_tokens": 300,
        "top_k": None,
        "top_p": 1.0,
        "do_sample": True,  
        "num_return_sequences": 100,
        "num_beams": 1,
        "return_dict_in_generate":True,
        "output_scores":True
    }
}

contrastive_generation_config_od99 = {
    "name": "contrastive_decoding_greedy",
    "multiple_rounds_generation": False,
    "student_model": "/auto/home/menuab/code/checkpoints/8c311987db124d9e87fc26da/125m_24k_8c31/",
    "expert_model": "/auto/home/menuab/code/checkpoints/0d992caa5ec443d9aefc289c/125m_256k_0d99/",
    "config": {
        "eos_token_id": 20,
        "max_length": 300,
        "st_coef": .2,
        "student_temperature": 1.,
        "num_beams": 1,
        "adaptability_constant": 1,
        "return_dict_in_generate": True,
        "output_scores": True,
        "num_return_sequences": 1,
        "do_sample": False,
        "student_min_prob": 0.0,
        "contrastive_decoding": "student",
        "use_cache": True,
    }
}

contrastive_generation_config_fe31 = {
    "name": "contrastive_decoding_greedy",
    "multiple_rounds_generation": False,
    "student_model": "/auto/home/menuab/code/checkpoints/fe31d8c5edfd4b93b72f1b60/125m_120k_fe31/",
    "expert_model": "/auto/home/menuab/code/checkpoints/fe31d8c5edfd4b93b72f1b60/125m_512k_fe31/",
    "config": {
        "eos_token_id": 20,
        "max_length": 300,
        "st_coef": .2,
        "student_temperature": 1.,
        "num_beams": 1,
        "adaptability_constant": 1,
        "return_dict_in_generate": True,
        "output_scores": True,
        "num_return_sequences": 1,
        "do_sample": False,
        "student_min_prob": 0.0,
        "contrastive_decoding": "student",
        "use_cache": True,
    }
}

contrastive_generation_config_f2c6 = {
    "name": "contrastive_decoding_greedy",
    "multiple_rounds_generation": False,
    "student_model": "/auto/home/menuab/code/checkpoints/f2c6ebb289994595a478f513/125m_126k_f2c6/",
    "expert_model": "/auto/home/menuab/code/checkpoints/f2c6ebb289994595a478f513/125m_63k_f2c6/",
    "config": {
        "eos_token_id": 20,
        "max_length": 300,
        "st_coef": .2,
        "student_temperature": 1.,
        "num_beams": 1,
        "adaptability_constant": 1,
        "return_dict_in_generate": True,
        "output_scores": True,
        "num_return_sequences": 1,
        "do_sample": False,
        "student_min_prob": 0.0,
        "contrastive_decoding": "student",
        "use_cache": True,
    }
}

contrastive_generation_config_9075 = {
    "name": "contrastive_decoding_greedy",
    "multiple_rounds_generation": False,
    "student_model": "/auto/home/menuab/code/checkpoints/90758da0b8564bae8a14bbef/125m_24k_9075/",
    "expert_model": "/auto/home/menuab/code/checkpoints/90758da0b8564bae8a14bbef/125m_63k_9075/",
    "config": {
        "eos_token_id": 20,
        "max_length": 300,
        "st_coef": .2,
        "student_temperature": 1.,
        "num_beams": 1,
        "adaptability_constant": 1,
        "return_dict_in_generate": True,
        "output_scores": True,
        "num_return_sequences": 1,
        "do_sample": False,
        "student_min_prob": 0.0,
        "contrastive_decoding": "student",
    }
}

contrastive_generation_config_26d3 = {
    "name": "contrastive_decoding_greedy",
    "multiple_rounds_generation": False,
    "student_model": "/auto/home/menuab/code/checkpoints/26d322857a184fcbafda5d4a/125m_69k_26d3/",
    "expert_model": "/auto/home/menuab/code/checkpoints/26d322857a184fcbafda5d4a/125m_118k_26d3/",
    "config": {
        "eos_token_id": 20,
        "max_length": 300,
        "st_coef": .2,
        "student_temperature": 1.,
        "num_beams": 1,
        "adaptability_constant": 1,
        "return_dict_in_generate": True,
        "output_scores": True,
        "num_return_sequences": 1,
        "do_sample": False,
        "student_min_prob": 0.0,
        "contrastive_decoding": "student",
    }
}

model_125m_122k_9954 = "/auto/home/menuab/code/checkpoints/9954e52e400b43d18d3a40f6/125m_122k_9954"
model_125m_118k_26d3 = "/auto/home/menuab/code/checkpoints/26d322857a184fcbafda5d4a/125m_118k_26d3/"
model_125m_4k_b8cb = "/auto/home/menuab/code/checkpoints/b8cb3a81b61e40aa919e06bc/125m_4k_b8cb/"
model_125m_9k_8073 = "/auto/home/menuab/code/checkpoints/8073deb785f04fcd891e58db/125m_9k_8073/"
model_125m_126k_f3fb = "/auto/home/menuab/code/checkpoints/f3fbd012918247a388efa732/125m_126k_f3fb/"
model_125m_126k_f2c6 = "/auto/home/menuab/code/checkpoints/f2c6ebb289994595a478f513/125m_126k_f2c6/"
model_125m_63k_f2c6 = "/auto/home/menuab/code/checkpoints/f2c6ebb289994595a478f513/125m_63k_f2c6/"
model_125m_313k_cf98 = "/auto/home/menuab/code/checkpoints/cf982665b6c04c83a310b97d/125m_313k_cf98/"
model_125m_512k_fe31 = "/auto/home/menuab/code/checkpoints/fe31d8c5edfd4b93b72f1b60/125m_512k_fe31/"
model_125m_256k_0d99 = "/auto/home/menuab/code/checkpoints/0d992caa5ec443d9aefc289c/125m_256k_0d99/"
model_1b_131k_d5c2   = "/auto/home/menuab/code/checkpoints/d5c2c8db3c554447a27697bf/1.3b_131k_d5c2/"
model_125m_73k_assay_87dc = "/auto/home/menuab/code/checkpoints/87dc7180e49141deae4ded57/125m_73k_assay_87dc/"
model_125m_73k_assay_c6af = "/auto/home/menuab/code/checkpoints/c6af41c79f1244f698cc1153/125m_73k_assay_c6af"
model_125m_18k_a37d = "/nfs/dgx/raid/chem/checkpoints/facebook/galactica-125m/a37d0362e15c4c969307aef8/checkpoint-18432"
model_125m_20k_6913 = "/nfs/dgx/raid/chem/checkpoints/facebook/galactica-125m/6913ba7695b040c597741e76/checkpoint-20480"
model_2b_11k_5292 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/52924785fbfc4c2e839d7e43/2b_11k_5292"
model_2b_12k_5292 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/52924785fbfc4c2e839d7e43/2b_12k_5292"
model_2b_20k_c60e = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/c60eb72c819147acb8a45cee/2b_20k_c60e/"
model_2b_20k_869e = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/869e097219da4a4fbbadcc11/checkpoint-20000"
model_2b_20k_dbf4 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/dbf4eca0f4234b97b2894278/checkpoint-20000"
model_2b_20k_9283 = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/92831d9b0ba24115ad3d2b1e/checkpoint-40000"

gemma_tokenizer_path = "/auto/home/menuab/code/ChemLactica/chemlactica/tokenizer/GemmaTokenizer"
galactica_tokenizer_path =         "/auto/home/menuab/code/ChemLacticaTestSuite/src/tokenizer/galactica-125m/"
chemlactica_tokenizer_50028_path = "/auto/home/menuab/code/ChemLacticaTestSuite/src/tokenizer/ChemLacticaTokenizer_50028"
chemlactica_tokenizer_50066_path = "/auto/home/menuab/code/ChemLacticaTestSuite/src/tokenizer/ChemLacticaTokenizer_50066"

top_N = 100
n_per_vs_rmse = 4
regexp = "^.*?(?=\\[END_SMILES])"
# torch_dtype = "float32"
torch_dtype = "bfloat16"
# device = "cuda:1"
device = "cuda:0"
# device = 'cpu'
# target_dist = "prior"
target_dist = "uniform"
std_var = 1

target_dists = ["prior", "uniform"]
models = [model_2b_11k_5292, model_2b_12k_5292, model_2b_20k_dbf4]
gen_configs = [greedy_generation_config, greedy_beam6_generation_config, contrastive_generation_config_26d3, greedy_beam_generation_config]

evaluation_config = {
    "test_suite":            test_suite,
    "property_range":        property_range,
    "generation_config":     greedy_generation_config,
    "model_checkpoint_path": model_2b_20k_869e,
    "tokenizer_path":        gemma_tokenizer_path,
    "std_var":               0,
    "torch_dtype":           torch_dtype,
    "device":                device,
    "regexp":                regexp,
    "top_N":                 top_N,
    "target_dist":           target_dist,
    "n_per_vs_rmse":         n_per_vs_rmse,
    "include_eos":           True,
    "include_start_smiles":  False,
    "check_for_novelty":     True,
    "track":                 True,
    "plot":                  True,
    "description":           ""
}
if 'gemma' in evaluation_config['model_checkpoint_path']:
    evaluation_config['generation_config']['config']['eos_token_id'] = 8

evaluation_config["description"] = f'{evaluation_config["model_checkpoint_path"].split("/")[-1]},'\
    f'{evaluation_config["generation_config"]["name"]},CoT:{not evaluation_config["include_start_smiles"]}'
evaluation_configs = [evaluation_config]


# evaluation_configs = []
# for model in models:
#     conf = copy.deepcopy(evaluation_config)
#     conf['model_checkpoint_path'] = model
#     conf['generation_config']['config']['eos_token_id'] = 8
#     conf["description"] = f'{conf["model_checkpoint_path"][-15:-1]},{conf["target_dist"]},'\
#         f'{conf["generation_config"]["name"]}'
#     evaluation_configs.append(conf)

# evaluation_config2 = {
#     "test_suite":            test_suite,
#     "property_range":        property_range,
#     "generation_config":     greedy_generation_config,
#     "model_checkpoint_path": model_125m_126k_f2c6,
#     "tokenizer_path":        chemlactica_tokenizer_50066_path,
#     "torch_dtype":           torch_dtype,
#     "device":                device,
#     "regexp":                regexp,
#     "top_N":                 top_N,
#     "n_per_vs_rmse":         n_per_vs_rmse,
#     "include_eos":           True,
#     "include_start_smiles":  True,
#     "check_for_novelty":     True,
#     "track":                 True,
#     "plot":                  True,
#     "description":           ""
# }
# evaluation_config2["description"] = f'{evaluation_config2["model_checkpoint_path"][-15:-1]},'\
#     f'{evaluation_config2["generation_config"]["name"]},CoT:{not evaluation_config2["include_start_smiles"]}'


# num_beams = [6,8,10,12]
# length_penalty = [-7,-9,-11,-13,-20] # only with num_beams>1
# repetition_penalty = [1.0]
# num_beam_groups = [2,3,4,5,6]
# diversity_penalty = [0.5,0.8,1.0 ]# only for num_beam_groups > 1
# renormalize_logits = True
# evaluation_configs = []
# for nb in num_beams:
#     for lp in length_penalty:
#         for rp in repetition_penalty:
#             for nbg in range(2, nb):
#                 for dp in diversity_penalty:
#                     if nb % nbg == 0:
#                         conf = copy.deepcopy(evaluation_config)
#                         conf["generation_config"]["config"]["num_beams"] = nb
#                         conf["generation_config"]["config"]["num_return_sequences"] = nb
#                         conf["generation_config"]["config"]["length_penalty"] = lp
#                         conf["generation_config"]["config"]["repetition_penalty"] = rp
#                         conf["generation_config"]["config"]["num_beam_groups"] = nbg
#                         conf["generation_config"]["config"]["diversity_penalty"] = dp
#                         conf["generation_config"]["description"] = f"{nb=},{lp=},{rp=},{nbg=},{dp=}"
#                         evaluation_configs.append(conf)
#                         del conf
#                         # print(conf["generation_config"]["config"])