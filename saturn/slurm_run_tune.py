import submitit
import itertools as it
import datetime
import yaml
import os
import copy


def create_hparam_configs(config_file_path):
    config_tune = yaml.safe_load(open("./hparams_tune.yaml"))
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
        config = copy.deepcopy(config_default)
        for i, p in enumerate(params):
            if '+' in hparam_names[i]:
                a, b = hparam_names[i].split("+")
                config[a][b] = p
            else:
                config[hparam_names[i]] = p
        all_configs.append(config)
    
    return all_configs


if __name__ == "__main__":
    # seeds = [0, 1, 2]
    seeds = [3, 4]

    config_file_path = "./chemlactica_125m_hparams.yaml"
    # config_file_path = "./chemlactica_1.3b_hparams.yaml"
    # config_file_path = "./chemma_2b_hparams.yaml"
    hparam_configs = create_hparam_configs(config_file_path)
    model_name = "-".join(config_file_path.split("/")[-1].split("_")[:2])

    executor = submitit.AutoExecutor(folder="slurm_jobs/saturn-tune/job_%j")
    executor.update_parameters(
        name="chemlactica-saturn-tune", timeout_min=15,
        gpus_per_node=1,
        nodes=1, mem_gb=30, cpus_per_task=1,
        slurm_array_parallelism=10
    )
    print(len(hparam_configs))
    
    jobs = []
    with executor.batch():
        for i, config in enumerate(hparam_configs):
            formatted_date_time = datetime.datetime.now().strftime("%Y-%m-%d")
            base = f"results/{formatted_date_time}"
            os.makedirs(base, exist_ok=True)
            v = 0
            name = model_name + "-" + "+".join(config["strategy"])
            while os.path.exists(os.path.join(base, f"{name}-{v}")):
                v += 1
            output_dir = os.path.join(base, f"{name}-{i}")
            os.makedirs(output_dir, exist_ok=True)
            yaml.safe_dump(config, open(os.path.join(output_dir, "hparams.yaml"), "w"))
            for seed in seeds:
                function = submitit.helpers.CommandFunction([
                    'python3', 'main.py',
                    '--config_default', os.path.join(output_dir, "hparams.yaml"),
                    '--output_dir', output_dir,
                    '--seed', str(seed),
                    "--illustrative", "True"
                ])
                print(' '.join(function.command))
                job = executor.submit(function)
                jobs.append(job)
