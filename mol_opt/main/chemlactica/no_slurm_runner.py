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


if __name__ == "__main__":
    task_names = get_hparam_tunning_tasks()
    # task_names = get_ablation_tasks()
    # task_names = get_all_tasks()
    n_runs = 3

    config_file_path = "main/chemlactica/chemlactica_125m_hparams.yaml"
    # config_file_path = "main/chemlactica/chemlactica_1.3b_hparams.yaml"
    hparam_configs = create_hparam_configs(config_file_path)
    # infer_config = [yaml.safe_load(open(config_file_path))]
    model_name = "-".join(config_file_path.split("/")[-1].split("_")[:2])

    config_task_pairs = list(it.product(hparam_configs, task_names))
    ind = 0
    while ind < len(config_task_pairs):
        free_gpu_id = None
        for gpu_id in range(0, 8):
            is_free = True
            # print(f"Checking gpu {gpu_id}")
            for _ in range(5):
                if not is_gpu_free(gpu_id):
                    is_free = False
                    break
                time.sleep(2)

            if is_free:
                print(f"{gpu_id} is free")
                free_gpu_id = gpu_id
                break
            else:
                # print(f"{gpu_id} is buzy")
                pass

        if free_gpu_id is not None:
            print(f"putting job on gpu {free_gpu_id}")
            config, task_name = config_task_pairs[ind]
            ind += 1
            executor = submitit.LocalExecutor(folder="/auto/home/tigranfahradyan/slurm_jobs/PMO/job_%j")
            executor.update_parameters(
                name="chemlactica-pmo", timeout_min=int(n_runs * 3 * 60),
                gpus_per_node=1,
                visible_gpus=[free_gpu_id],
                nodes=1, mem_gb=50, cpus_per_task=1,
                slurm_array_parallelism=10
            )
            jobs = []
            formatted_date_time = datetime.datetime.now().strftime("%Y-%m-%d")
            base = f"main/chemlactica/results/{formatted_date_time}-tune-1"
            os.makedirs(base, exist_ok=True)
            v = 0
            name = model_name + "-" + "+".join(config["strategy"])
            while os.path.exists(os.path.join(base, f"{name}-{v}")):
                v += 1
            output_dir = os.path.join(base, f"{name}-{v}")
            # output_dir = "main/chemlactica/results/2024-05-11/chemlactica-125m-rej-sample-4"
            os.makedirs(output_dir, exist_ok=True)
            yaml.safe_dump(config, open(os.path.join(output_dir, "hparams.yaml"), "w"))
            function = submitit.helpers.CommandFunction([
                'python3', 'run.py',
                'chemlactica', '--oracles', task_name,
                '--task', 'production', '--n_runs', str(n_runs),
                '--config_default', os.path.join(output_dir, "hparams.yaml"),
                '--output_dir', output_dir,
            ])
            print(' '.join(function.command))
            # subprocess.run(function.command)
            job = executor.submit(function)
            print(job.job_id)
            jobs.append(job)
            time.sleep(30)