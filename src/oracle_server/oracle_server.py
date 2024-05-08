from flask import Flask, request, jsonify,Blueprint
import submitit
import argparse
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
NUM_CPU = None # None defaults to automatic allocation of all available CPU cores if possible


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
            oracle_params = {"vina_params":getattr(VinaConfigEnum, oracle_name.upper()).value,"vina_binary_path":VINA_PATH,"num_cpu":NUM_CPU}
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

def run_oracle_server(args):
    global VINA_PATH
    global NUM_CPU

    VINA_PATH = args.vina_path
    NUM_CPU = args.num_cpu
    app.run(port=5006,debug=False,host='0.0.0.0')

def my_random_function(args):
    print("my_random_function works")
    print(args.num_cpu)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Oracle server")
    parser.add_argument("--vina_path", type=str, required=True, help="Path to the AutoDock Vina binary")
    parser.add_argument("--num_cpu", type=int, required=False, help="Number of CPUs to use for Vina docking",default = None)
    args = parser.parse_args()

    slurm_params = {
        "slurm_job_name": "running_oracle_server",
        "timeout_min": 3,
        "nodes": 1,
        "tasks_per_node": 1,
        "cpus_per_task":1,
        "mem_gb": 0.5,
        "stderr_to_stdout": True,
    }

    executor = submitit.AutoExecutor(folder="/mnt/sxtn2/chem/oracle_server_logs/")
    executor.update_parameters(**slurm_params)
    print('before submit')
    job = executor.submit(my_random_function,args)
    print(job.result())

    #run_oracle_server(args)
