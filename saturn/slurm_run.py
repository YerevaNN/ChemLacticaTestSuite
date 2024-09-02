import os
import copy
import yaml
import submitit
import datetime


def get_all_targets():
    return [
        "fa7",
        "parp1",
        "5ht1b",
        "jak2",
        "braf",
    ]

def get_all_seeds():
    return list(range(10)) # 0 to 9


if __name__ == "__main__":
    target_proteins = get_all_targets()
    all_seeds = get_all_seeds()
    model_name = "chemlactica_1.3b"

    config_file_path = f"{model_name}_hparams.yaml"
    default_config = yaml.safe_load(open(config_file_path))

    configs_per_target = []
    for target in target_proteins:
        config = copy.deepcopy(default_config)
        config["target"] = target
        configs_per_target.append(config)

    executor = submitit.AutoExecutor(folder="slurm_jobs/saturn/job_%j")
    executor.update_parameters(
        name="chemlactica-against-saturn", timeout_min=int(len(target_proteins) * len(all_seeds) * 30),
        gpus_per_node=1,
        nodes=1, mem_gb=30, cpus_per_task=60,
        slurm_array_parallelism=4
    )
    
    jobs = []
    with executor.batch():
        for config in configs_per_target:
            formatted_date_time = datetime.datetime.now().strftime("%Y-%m-%d")
            base = f"results/{formatted_date_time}"
            os.makedirs(base, exist_ok=True)
            name = f"{model_name}"
            v = 0
            while os.path.exists(os.path.join(base, f"{name}-{v}")):
                v += 1
            output_dir = os.path.join(base, f"{name}-v{v}")
            os.makedirs(output_dir, exist_ok=True)

            for seed in all_seeds:
                config_with_seed = copy.deepcopy(config)
                config_with_seed["seed"] = seed
                hparams_path = os.path.join(output_dir, f"hparams-{config['target']}-seed{seed}.yaml")
                yaml.safe_dump(config_with_seed, open(hparams_path, "w"))
                 
                function = submitit.helpers.CommandFunction([
                    'python3', 'main.py',
                    '--config_default', hparams_path,
                    '--output_dir', output_dir,
                    '--seed', str(seed),
                    '--target', config['target'],
                ])
                print(' '.join(function.command))
                job = executor.submit(function)
                jobs.append(job)
