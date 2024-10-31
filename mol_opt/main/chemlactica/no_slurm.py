import submitit
import subprocess
import itertools as it
import datetime
import yaml
import os
import copy
import time


def create_hparam_configs(config_file_path):
    config_tune = yaml.safe_load(open("main/chemlactica/hparams_tune.yaml"))
    config_merged = {}
    for key, value in config_tune["parameters"].items():
        if type(value) == list:
            config_merged[key] = value
        else:
            for k, v in value.items():
                config_merged[key+'+'+k] = v
    
    config_default = yaml.safe_load(open(config_file_path))
    hparam_names = list(config_merged.keys())
    all_configs = []
    for params in it.product(*config_merged.values()):
        # pprint(params)
        # pprint(hparam_names)
        config = copy.deepcopy(config_default)
        for i, p in enumerate(params):
            if '+' in hparam_names[i]:
                a, b = hparam_names[i].split("+")
                config[a][b] = p
            else:
                config[hparam_names[i]] = p
        # pprint(params)
        # pprint(config)
        all_configs.append(config)
        # print(config)
    return all_configs


def get_all_tasks():
    return [
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


def get_ablation_tasks():
    return [
        "jnk3",
        "median1",
        "sitagliptin_mpo",
        "scaffold_hop"
    ]


def get_hparam_tunning_tasks():
    return [
        "perindopril_mpo",
        "zaleplon_mpo"
    ]


def check_gpu_usage(gpu_id):
    try:
        # Run the nvidia-smi command
        cmd = ['nvidia-smi','-i',f"{gpu_id}"]
        output = subprocess.check_output(cmd)
        output = output.decode('utf-8')
        if "No running processes found" in output:
            return True
        else:
            return False

    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi command: {e}")


def is_gpu_free(gpu_id):
    try:
        # Run the nvidia-smi command
        cmd = ['nvidia-smi','-i',f"{gpu_id}"]
        output = subprocess.check_output(cmd)
        output = output.decode('utf-8')
        if "No running processes found" in output:
            return True
        else:
            return False

    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi command: {e}")


def is_gpu_free(gpu_id):
    try:
        # Run the nvidia-smi command
        cmd = ['nvidia-smi','-i',f"{gpu_id}"]
        output = subprocess.check_output(cmd)
        output = output.decode('utf-8')
        if "No running processes found" in output:
            return True
        else:
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi command: {e}")
        return False


if __name__ == "__main__":

    commands = [
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-0/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-0",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-0/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-0",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-1/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-1",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-1/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-1",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-2/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-2",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-2/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-2",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-3/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-3",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-3/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-3",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-4/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-4",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-4/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-4",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-5/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-5",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-5/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-5",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-6/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-6",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-6/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-6",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-7/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-7",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-7/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-7",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-8/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-8",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-8/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-8",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-9/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-9",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-9/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-9",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-10/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-10",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-10/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-10",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-11/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-11",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-11/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-11",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-12/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-12",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-12/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-12",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-13/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-13",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-13/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-13",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-14/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-14",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-14/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-14",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-15/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-15",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-15/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-15",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-16/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-16",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-16/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-16",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-17/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-17",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-17/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-17",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-18/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-18",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-18/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-18",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-19/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-19",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-19/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-19",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-20/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-20",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-20/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-20",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-21/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-21",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-21/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-21",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-22/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-22",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-22/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-22",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-23/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-23",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-23/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-23",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-24/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-24",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-24/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-24",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-25/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-25",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-25/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-25",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-26/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-26",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-26/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-26",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-27/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-27",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-27/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-27",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-28/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-28",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-28/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-28",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-29/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-29",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-29/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-29",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-30/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-30",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-30/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-30",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-31/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-31",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-31/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-31",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-32/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-32",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-32/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-32",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-33/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-33",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-33/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-33",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-34/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-34",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-34/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-34",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-35/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-35",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-35/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-35",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-36/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-36",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-36/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-36",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-37/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-37",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-37/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-37",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-38/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-38",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-38/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-38",
        # "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-39/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-39",
        # "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-39/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-39",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-40/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-40",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-40/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-40",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-41/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-41",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-41/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-41",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-42/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-42",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-42/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-42",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-43/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-43",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-43/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-43",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-44/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-44",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-44/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-44",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-45/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-45",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-45/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-45",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-46/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-46",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-46/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-46",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-47/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-47",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-47/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-47",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-48/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-48",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-48/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-48",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-49/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-49",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-49/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-49",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-50/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-50",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-50/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-50",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-51/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-51",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-51/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-51",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-52/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-52",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-52/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-52",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-53/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-53",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-53/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-53",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-54/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-54",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-54/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-54",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-55/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-55",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-55/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-55",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-56/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-56",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-56/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-56",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-57/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-57",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-57/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-57",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-58/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-58",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-58/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-58",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-59/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-59",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-59/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-59",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-60/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-60",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-60/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-60",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-61/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-61",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-61/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-61",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-62/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-62",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-62/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-62",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-63/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-63",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-63/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-63",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-64/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-64",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-64/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-64",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-65/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-65",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-65/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-65",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-66/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-66",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-66/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-66",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-67/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-67",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-67/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-67",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-68/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-68",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-68/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-68",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-69/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-69",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-69/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-69",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-70/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-70",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-70/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-70",
        "python3 run.py chemlactica --oracles perindopril_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-71/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-71",
        "python3 run.py chemlactica --oracles zaleplon_mpo --task production --n_runs 3 --config_default main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-71/hparams.yaml --output_dir main/chemlactica/results/2024-06-10-tune/chemlactica-125m-rej-sample-v2-71",
    ]

    executor = submitit.AutoExecutor(folder="/auto/home/tigranfahradyan/slurm_jobs/PMO/job_%j")
    executor.update_parameters(
        name="chemlactica-pmo", timeout_min=int(3 * 60),
        gpus_per_node=1,
        nodes=1, mem_gb=50, cpus_per_task=2,
        slurm_array_parallelism=10
    )
    jobs = []
    with executor.batch():
        for command in commands:
            function = submitit.helpers.CommandFunction(command.split(" "))
            print(' '.join(function.command))
            # subprocess.run(function.command)
            job = executor.submit(function)
            jobs.append(job)

    # ind = 0
    # while ind < len(commands):
    #     free_gpu_index = None
    #     for i in range(0, 8):
    #         is_free = True
    #         for _ in range(10):
    #             if not is_gpu_free(i):
    #                 is_free = False
    #                 break
    #             time.sleep(1)
    #         if is_free:
    #             print(f"gpu {i} is free")
    #             free_gpu_index = i
    #             break
    #         else:
    #             # print(f"gpu {i} is buzy")
    #             pass
    #     if free_gpu_index != None:
    #         print(f"putting job on {free_gpu_index}")
    #         cur_commands = commands[ind:ind + 2]
    #         ind += 2
    #         print(ind)
    #         executor = submitit.LocalExecutor(folder="/auto/home/tigranfahradyan/slurm_jobs/PMO/job_%j")
    #         executor.update_parameters(
    #             name="chemlactica-pmo", timeout_min=int(3 * 3 * 60),
    #             gpus_per_node=1,
    #             visible_gpus=[free_gpu_index],
    #             nodes=1, mem_gb=50, cpus_per_task=1,
    #             slurm_array_parallelism=10
    #         )
    #         jobs = []
    #         with executor.batch():
    #             for command in cur_commands:
    #                 function = submitit.helpers.CommandFunction(command.split(" "))
    #                 print(' '.join(function.command))
    #                 # subprocess.run(function.command)
    #                 job = executor.submit(function)
    #                 jobs.append(job)

    #         for job in jobs:
    #             print(job.job_id)
    #         time.sleep(10)