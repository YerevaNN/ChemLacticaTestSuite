from flask import Flask, request, jsonify
from functools import cache

import sys
sys.path.append('saturn')
from tdc import Oracle
from saturn.oracles.oracle import Oracle as SaturnOracle, OracleConfiguration
from saturn.oracles.dataclass import OracleComponentParameters
from saturn.diversity_filter.diversity_filter import DiversityFilter, DiversityFilterParameters

app = Flask(__name__)


@cache
def get_illustrative_exp_oracle_and_div_filter(max_oracle_calls):
    tpsa = OracleComponentParameters(
        name="tpsa",
        specific_parameters={},
        reward_shaping_function_parameters={
            "transformation_function": "sigmoid",
            "parameters": {"low": 75, "high": 350, "k": 0.15},
        },
    ).__dict__
    mw = OracleComponentParameters(
        name="mw",
        specific_parameters={},
        reward_shaping_function_parameters={
            "transformation_function": "double_sigmoid",
            "parameters": {
                "low": 0,
                "high": 350,
                "coef_div": 500,
                "coef_si": 250,
                "coef_se": 250,
            },
        },
    ).__dict__
    num_rings = OracleComponentParameters(
        name="num_rings",
        specific_parameters={},
        reward_shaping_function_parameters={
            "transformation_function": "step",
            "parameters": {"low": 2, "high": 5},
        },
    ).__dict__
    config = OracleConfiguration(
        budget=max_oracle_calls,
        aggregator="product",
        allow_oracle_repeats=False,
        components=[tpsa, mw, num_rings],
    )
    diversity_filter = DiversityFilter(DiversityFilterParameters())
    ill_oracle = SaturnOracle(config)
    return ill_oracle, diversity_filter


@app.route('/illustrative', methods=['POST'])
def send_illustrative():
    data = request.json  # Assuming you are sending JSON data in the request body
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid data format.'}), 400

    smiles = data['smiles']
    max_oracle_calls = data['max_oracle_calls']
    
    ill_oracle, diversity_filter = get_illustrative_exp_oracle_and_div_filter(max_oracle_calls)
    smiles, scores = ill_oracle(smiles, diversity_filter=diversity_filter)
    return jsonify({'smiles': list(smiles), 'scores': list(scores)}), 200


@cache
def get_pmo_oracle(oracle_name):
    return Oracle(oracle_name)


@app.route(f"/pmo", methods=['POST'])
def dynamic_route():
    data = request.json  # Assuming you are sending JSON data in the request body
    if not isinstance(data, dict):
        return jsonify({'error': 'Expected a dict of oracle_name and smiles'}), 400 
    
    oracle_name = data['oracle_name']
    smiles = data['smiles']

    pmo_oracle = get_pmo_oracle(oracle_name)
    return jsonify({'scores': pmo_oracle(smiles)}), 200


@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to illustrative and PMO experiments'}), 200


if __name__ == "__main__":
    pmo_tasks = [
        "albuterol_similarity",
        "amlodipine_mpo",
        "celecoxib_rediscovery",
        "deco_hop",
        "drd2",
        "fexofenadine_mpo",
        "gsk3b",
        "isomers_c7h8n2o2",
        "isomers_c9h10n2o2pf2cl",
        "jnk3",
        "median1",
        "median2",
        "mestranol_similarity",
        "osimertinib_mpo",
        "perindopril_mpo",
        "qed",
        "ranolazine_mpo",
        "scaffold_hop",
        "sitagliptin_mpo",
        "thiothixene_rediscovery",
        "troglitazone_rediscovery",
        "valsartan_smarts",
        "zaleplon_mpo"
    ]
    for task in pmo_tasks:
        get_pmo_oracle(task)

    app.run(debug=False, port=5400, host="0.0.0.0", processes=8, threaded=False)