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
    }
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
        "mean": 2.8,
    },
    "qed": {
        "range": (0.01, 1),
        "step":  0.01,
        "mean": 0.75,
    },
    "weight": {
        "range": (100.1, 1000),
        "step":  1,
        "mean": 290,
    },
    "clogp": {
        "range": (1.1, 10),
        "step":  0.1,
        "mean": 3,
    }
}  

mock_property_range = {
    "clogp": {
        "range": (3.2, 10),
        "step":  1,
        "mean": 3,
    },
    "sas": {
        "range": (1.2, 10),
        "step":  1,
        "mean": 2.8,
    },
    "qed": {
        "range": (0.01, 1),
        "step":  .1,
        "mean": 0.75,
    },
    "weight": {
        "range": (104.1, 1000),
        "step":  200,
        "mean": 290,
    }
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
    "name": "greedy_beam=5",
    "multiple_rounds_generation": False,
    "config": {
        "eos_token_id": 20,
        "max_new_tokens": 300,
        "do_sample": False,  
        "num_return_sequences": 1,
        "num_beams": 5,
        "return_dict_in_generate":True,
        "output_scores":True
    }
}

nongreedy_generation_config = {
    "name": "nongreedy_5of20",
    "top_N": 5,
    "multiple_rounds_generation": False,
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
    "student_model": "/home/menuab/code/checkpoints/8c311987db124d9e87fc26da/125m_24k_8c31/",
    "expert_model": "/home/menuab/code/checkpoints/0d992caa5ec443d9aefc289c/125m_256k_0d99/",
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

contrastive_generation_config_fe31 = {
    "name": "contrastive_decoding_greedy",
    "multiple_rounds_generation": False,
    "student_model": "/home/menuab/code/checkpoints/fe31d8c5edfd4b93b72f1b60/125m_120k_fe31/",
    "expert_model": "/home/menuab/code/checkpoints/fe31d8c5edfd4b93b72f1b60/125m_512k_fe31/",
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

top_N = 100
n_per_vs_rmse = 4
regexp = "^.*?(?=\\[END_SMILES])"

model_125m_512k_fe31 = "/home/menuab/code/checkpoints/fe31d8c5edfd4b93b72f1b60/125m_512k_fe31/"
model_125m_249k_0d99 = "/home/menuab/code/checkpoints/0d992caa5ec443d9aefc289c/125m_249k_0d99/"
model_125m_253k_0d99 = "/home/menuab/code/checkpoints/0d992caa5ec443d9aefc289c/125m_253k_0d99/"
model_125m_256k_0d99 = "/home/menuab/code/checkpoints/0d992caa5ec443d9aefc289c/125m_256k_0d99/"
model_125m_253k_ac79 = "/home/menuab/code/checkpoints/ac7915df73b24ee3a4e172d6/125m_253k_ac79/"
model_125m_241k_ac79 = "/home/menuab/code/checkpoints/ac7915df73b24ee3a4e172d6/125m_241k_ac79/"
model_1b_131k_d5c2   = "/home/menuab/code/checkpoints/d5c2c8db3c554447a27697bf/1.3b_131k_d5c2/"
model_125m_73k_assay_87dc = "/home/menuab/code/checkpoints/87dc7180e49141deae4ded57/125m_73k_assay_87dc/"
model_125m_73k_assay_c6af = "/home/menuab/code/checkpoints/c6af41c79f1244f698cc1153/125m_73k_assay_c6af"
galactica_tokenizer_path =         "/home/menuab/code/ChemLacticaTestSuite/src/tokenizer/galactica-125m/"
chemlactica_tokenizer_50028_path = "/home/menuab/code/ChemLacticaTestSuite/src/tokenizer/ChemLacticaTokenizer_50028"
chemlactica_tokenizer_50066_path = "/home/menuab/code/ChemLacticaTestSuite/src/tokenizer/ChemLacticaTokenizer_50066"
# torch_dtype = "float32"
torch_dtype = "bfloat16"
device = "cuda:1"
# device = "cuda:0"
# device = 'cpu'

models = [model_125m_253k_ac79, model_125m_512k_fe31, model_125m_256k_0d99]
gen_configs = [nongreedy_calibration_generation_config]

evaluation_config = {
    "test_suite":            test_suite,
    "property_range":        property_range,
    "generation_config":     contrastive_generation_config_fe31,
    "model_checkpoint_path": model_125m_512k_fe31,
    "tokenizer_path":        chemlactica_tokenizer_50028_path,
    "torch_dtype":           torch_dtype,
    "device":                device,
    "regexp":                regexp,
    "top_N":                 top_N,
    "n_per_vs_rmse":         n_per_vs_rmse,
    "include_eos":           True,
    "check_for_novelty":     False,
    "track":                 False,
    "plot":                  True,
    "description": f"125m_512k_fe31-2xdata-CD-greedy",
}

# evaluation_config = {
#     "test_suite":            test_suite,
#     "property_range":        property_range,
#     "generation_config":     contrastive_generation_config,
#     "model_checkpoint_path": model_125m_512k_fe31,
#     "tokenizer_path":        chemlactica_tokenizer_50028_path,
#     "torch_dtype":           torch_dtype,
#     "device":                "cuda:0",
#     "regexp":                regexp,
#     "top_N":                 top_N,
#     "n_per_vs_rmse":         n_per_vs_rmse,
#     "include_eos":           True,
#     "check_for_novelty":     True,
#     "track":                 True,
#     "plot":                  True,
#     "description": f"CD-0d99-greedy",
# }
# evaluation_configs = []
# for gen_config in gen_configs:
#     for model in models:
#     # model = model_1b_131k_d5c2
#         conf = copy.copy(evaluation_config)
#         # conf["tokenizer_path"] = galactica_tokenizer_path
#         conf["model_checkpoint_path"] = model
#         conf["generation_config"] = gen_config
#         conf["description"] = f"{model.split('/')[-2]}_{gen_config['name']}"
#         evaluation_configs.append(conf)

evaluation_configs = [evaluation_config]