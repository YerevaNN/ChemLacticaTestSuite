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
    "qed": {
        "input_properties": ["qed"],
        "target_properties": ["qed"]
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