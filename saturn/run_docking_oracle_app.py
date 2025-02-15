from flask import Flask, request, jsonify
from rdkit import Chem
from rdkit.Chem import QED
from functools import cache

import sys
sys.path.append('saturn')
from saturn.oracles.docking.geam_oracle import GEAMOracle
from saturn.oracles.dataclass import OracleComponentParameters
from saturn.oracles.synthesizability.sascorer import calculateScore

app = Flask(__name__)


def get_oracle_params(target):
    return OracleComponentParameters(
        name="geam",
        specific_parameters={
            "target": target,
        },
        reward_shaping_function_parameters={
            "transformation_function": "sigmoid",
            "parameters": {"low": 75, "high": 350, "k": 0.15},
        },
    )


def get_targets():
    return [
        "fa7",
        "parp1",
        "5ht1b",
        "jak2",
        "braf",
    ]


@cache
def get_docking_oracles():
    return {
        target: GEAMOracle(get_oracle_params(target)) for target in get_targets()
    }


@app.route('/fa7', methods=['POST'])
def send_fa7():
    data = request.json  # Assuming you are sending JSON data in the request body
    if not isinstance(data, list):
        return jsonify({'error': 'Expected a list'}), 400
    
    rdkit_mols = [Chem.MolFromSmiles(s) for s in data]
    raw_vina_scores, *_, aggregated_scores = get_docking_oracles()["fa7"](rdkit_mols)
    sa_scores = [calculateScore(m) for m in rdkit_mols]
    qed_scores = [QED.qed(m) for m in rdkit_mols]
    return jsonify({
        'vina_scores': list(raw_vina_scores),
        "scores": list(aggregated_scores),
        "sa_scores": sa_scores,
        "qed_scores": qed_scores
    }), 200


@app.route('/parp1', methods=['POST'])
def send_parp1():
    data = request.json  # Assuming you are sending JSON data in the request body
    if not isinstance(data, list):
        return jsonify({'error': 'Expected a list'}), 400
    
    rdkit_mols = [Chem.MolFromSmiles(s) for s in data]
    raw_vina_scores, *_, aggregated_scores = get_docking_oracles()["parp1"](rdkit_mols)
    sa_scores = [calculateScore(m) for m in rdkit_mols]
    qed_scores = [QED.qed(m) for m in rdkit_mols]
    return jsonify({
        'vina_scores': list(raw_vina_scores),
        "scores": list(aggregated_scores),
        "sa_scores": sa_scores,
        "qed_scores": qed_scores
    }), 200


@app.route('/5ht1b', methods=['POST'])
def send_5ht1b():
    data = request.json  # Assuming you are sending JSON data in the request body
    if not isinstance(data, list):
        return jsonify({'error': 'Expected a list'}), 400
    
    rdkit_mols = [Chem.MolFromSmiles(s) for s in data]
    raw_vina_scores, *_, aggregated_scores = get_docking_oracles()["5ht1b"](rdkit_mols)
    sa_scores = [calculateScore(m) for m in rdkit_mols]
    qed_scores = [QED.qed(m) for m in rdkit_mols]
    return jsonify({
        'vina_scores': list(raw_vina_scores),
        "scores": list(aggregated_scores),
        "sa_scores": sa_scores,
        "qed_scores": qed_scores
    }), 200


@app.route('/jak2', methods=['POST'])
def send_jak2():
    data = request.json  # Assuming you are sending JSON data in the request body
    if not isinstance(data, list):
        return jsonify({'error': 'Expected a list'}), 400
    
    rdkit_mols = [Chem.MolFromSmiles(s) for s in data]
    raw_vina_scores, *_, aggregated_scores = get_docking_oracles()["jak2"](rdkit_mols)
    sa_scores = [calculateScore(m) for m in rdkit_mols]
    qed_scores = [QED.qed(m) for m in rdkit_mols]
    return jsonify({
        'vina_scores': list(raw_vina_scores),
        "scores": list(aggregated_scores),
        "sa_scores": sa_scores,
        "qed_scores": qed_scores
    }), 200


@app.route('/braf', methods=['POST'])
def send_braf():
    data = request.json  # Assuming you are sending JSON data in the request body
    if not isinstance(data, list):
        return jsonify({'error': 'Expected a list'}), 400
    
    rdkit_mols = [Chem.MolFromSmiles(s) for s in data]
    raw_vina_scores, *_, aggregated_scores = get_docking_oracles()["braf"](rdkit_mols)
    sa_scores = [calculateScore(m) for m in rdkit_mols]
    qed_scores = [QED.qed(m) for m in rdkit_mols]
    return jsonify({
        'vina_scores': list(raw_vina_scores),
        "scores": list(aggregated_scores),
        "sa_scores": sa_scores,
        "qed_scores": qed_scores
    }), 200


@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'hello'}), 200


if __name__ == "__main__":
    app.run(debug=False, port=5455, host="0.0.0.0", processes=36, threaded=False)
