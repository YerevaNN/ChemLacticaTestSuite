from flask import Flask, request, jsonify,Blueprint
from vina_scoring import get_vina_score

from dataclasses import dataclass, asdict
from rd_pdbqt import MolToPDBQTBlock
from rdkit import Chem
import json
from rdkit.Chem import AllChem
from vina import Vina
from vina_config import VinaConfig, VinaConfigEnum
from typing import List, Union
from tdc import Oracle
from enum import Enum

app = Flask(__name__)
VINA_PATH = None

oracle_blueprint = Blueprint('oracles', __name__)

def get_request_data(request)->str:
    input_string = request.data.decode('utf-8')
    validate_input_string(input_string)
    input_data = json.loads(input_string)

    return input_data

def validate_input_string(input_string):
    if not isinstance(input_string, str):
        raise ValueError('input is not a string')
    elif input_string == '':
        raise ValueError('input_string is missing')
    elif input_string is None:
        raise ValueError('input_string is None')


def get_oracle(oracle_type,oracle_name):
    match oracle_type:
        case 'vina':
            oracle_params = {"vina_params":getattr(VinaConfigEnum, oracle_name.upper()).value}
            scoring_function = get_vina_score
        case 'tdc': # these endpoint allow for the use of any TDC oracle by name, see list at below url:
            # https://github.com/mims-harvard/TDC/blob/0469d9af0d4124490f3f8d922d6207b4e6dacabe/tdc/metadata.py#L894 
            scoring_function = Oracle(oracle_name) # Note that the Oracle object must be from tdc
            oracle_params = {} # no parameters needed for calling these oracles
        case _:
            return jsonify({'error': f'Invalid oracle type: {oracle_type}'}), 404
    return scoring_function, oracle_params


@oracle_blueprint.route('/<oracle_type>/<oracle_name>', methods=['POST'])
def get_oracle_score(oracle_type,oracle_name):

    try:
        scoring_function, oracle_params = get_oracle(oracle_type,oracle_name)
    except AttributeError:
        return jsonify({'error': f'Invalid oracle: {oracle_name}'}), 404

    try:
        input_data = get_request_data(request)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except json.JSONDecodeError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 501
    try:
        result = scoring_function(input_data,**oracle_params)
    except Exception as e:
        return jsonify({'error': str(e)}), 402
    return jsonify({'result': result}), 200

app.register_blueprint(oracle_blueprint, url_prefix='/oracles')

def main(vina_binary_path):
    global VINA_PATH
    VINA_PATH = vina_binary_path
    app.run(port=5006,debug=False,host='0.0.0.0')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Oracle server")
    parser.add_argument("--vina-path", required=True, help="Path to the AutoDock Vina binary")
    args = parser.parse_args()
    main(args.vina_path)
