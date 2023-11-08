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
        "range": (1, 10),
        "step":  0.1
    },
    "qed": {
        "range": (0.01, 1),
        "step":  0.01
    },
    "weight": {
        "range": (100.1, 1000),
        "step":  1
    },
    "clogp": {
        "range": (1, 10),
        "step":  0.1
    }
}  

mock_property_range = {
    "clogp": {
        "range": (1.2, 10),
        "step":  1
    },
    "sas": {
        "range": (1.2, 10),
        "step":  1
    },
    "qed": {
        "range": (0.01, 1),
        "step":  .1
    },
    "weight": {
        "range": (100.1, 1000),
        "step":  200
    }
}

greedy_generation_config = {
    "eos_token_id": 20,
    "max_length": 300,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "do_sample": False,  
    "num_return_sequences": 1,
    "num_beams": 1,
    "return_dict_in_generate":True,
    "output_scores":True
    }

greedy_beam_generation_config = {
    "eos_token_id": 20,
    "max_length": 300,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "do_sample": False,  
    "num_return_sequences": 1,
    "num_beams": 5,
    "return_dict_in_generate":True,
    "output_scores":True
    }

nongreedy_generation_config = {
    "eos_token_id": 20,
    "max_length": 400,
    "temperature": 1.0,
    "top_k": None,
    "top_p": 1.0,
    # "repetition_penalty": 1.0,
    "do_sample": True,  
    "num_return_sequences": 100,
    "num_beams": 1,
    "return_dict_in_generate":True,
    "output_scores":True
    }

contrastive_generation_config = {
    "eos_token_id": 20,
    "penalty_alpha" : 0.6,
    "top_k" : 4,
    "max_length" : 300,
    "num_beams": 1,
    "return_dict_in_generate":True,
    "output_scores":True,
    "num_return_sequences":500,
    # "num_beams":10,
    "do_sample": True,
}

top_N = 1000
n_per_vs_rmse = 4
regexp = "^.*?(?=\\[END_SMILES])"

# model_checkpoint_path = "/home/hrant/chem/tigran/ChemLactica/checkpoints/facebook/galactica-125m/ac7915df73b24ee3a4e172d6/checkpoint-253952"
model_125m_253k = "/home/menuab/code/checkpoints/125m_253k/"
model_125m_241k = "/home/menuab/code/checkpoints/125m_241k/"
model_1b_131k = "/home/menuab/code/checkpoints/1.3b_131k/"
model_125m_76k_new = "/home/menuab/code/checkpoints/87dc7180e49141deae4ded57/checkpoint-73728/"
model_125m_73k_assay = "/home/menuab/code/checkpoints/c6af41c79f1244f698cc1153/125m_73k_withassay"
galactica_tokenizer_path = "src/tokenizer/galactica-125m/"
chemlactica_tokenizer_50028_path = "src/tokenizer/ChemLacticaTokenizer_50028"
chemlactica_tokenizer_50066_path = "src/tokenizer/ChemLacticaTokenizer_50066"
# torch_dtype = "float32"
torch_dtype = "bfloat16"
device = "cuda:1"
# device = 'cpu'

evaluation_config = {
    "test_suite":            mock_test_suite,
    "property_range":        mock_property_range,
    "generation_config":     nongreedy_generation_config,
    "model_checkpoint_path": model_125m_253k,
    "tokenizer_path":        chemlactica_tokenizer_50028_path,
    "torch_dtype":           torch_dtype,
    "device":                device,
    "regexp":                regexp,
    "top_N":                 top_N,
    "n_per_vs_rmse":         n_per_vs_rmse,
    "include_eos":           True,
    "check_for_novelty":     False,
    "track":                 True,
}