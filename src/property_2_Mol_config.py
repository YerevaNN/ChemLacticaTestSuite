test_suit = {
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

mock_test_suit = {
    "weight": {
        "input_properties": ["weight"],
        "target_properties": ["weight"]
    }
}

property_range = {
    "sas": {
        "range": (1, 10),
        "step":  0.1
    },
    "qed": {
        "range": (0, 1),
        "step":  0.01
    },
    "weight": {
        "range": (100.1, 500),
        "step":  1
    },
    "clogp": {
        "range": (1, 5),
        "step":  0.1
    }
}  

mock_property_range = {
    "weight": {
        "range": (100.1, 300),
        "step":  1
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

nongreedy_generation_config = {
    "eos_token_id": 20,
    "max_length": 400,
    "temperature": 1.0,
    "top_k": None,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "do_sample": True,  
    "num_return_sequences": 20,
    "num_beams": 1,
    "return_dict_in_generate":True,
    "output_scores":True
    }

top_N = 5
regexp = "^.*?(?=\\[END_SMILES])"

# model_checkpoint_path = "/home/hrant/chem/tigran/ChemLactica/checkpoints/facebook/galactica-125m/ac7915df73b24ee3a4e172d6/checkpoint-253952"
model_125m_253k = "/home/menuab/code/checkpoints/125m_253k/"
model_125m_241k = "/home/menuab/code/checkpoints/125m_241k/"
model_1b_131k = "/home/menuab/code/checkpoints/1.3b_131k/"
galactica_tokenizer_path = "src/tokenizer/galactica-125m/"
chemlactica_tokenizer_path = "src/tokenizer/ChemLacticaTokenizer"
# torch_dtype = "float32"
torch_dtype = "bfloat16"
device = "cuda:0"

evaluation_config = {
    "test_suit":             test_suit,
    "property_range":        property_range,
    "generation_config":     greedy_generation_config,
    "model_checkpoint_path": model_125m_253k,
    "tokenizer_path":        chemlactica_tokenizer_path,
    "torch_dtype":           torch_dtype,
    "device":                device,
    "regexp":                regexp,
    "top_N":                 top_N,
    "generate_log_file":     True,
    "include_eos":           True,
}