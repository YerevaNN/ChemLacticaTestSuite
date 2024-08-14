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
    model_name = "chemlactica-125m"

    config_file_path = "default_params.yaml"
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
        nodes=1, mem_gb=30, cpus_per_task=10,
        slurm_array_parallelism=10
    )
    
    jobs = []
    with executor.batch():
        for config in configs_per_target:
            formatted_date_time = datetime.datetime.now().strftime("%Y-%m-%d")
            base = f"saturn/results/{formatted_date_time}"
            os.makedirs(base, exist_ok=True)

            for seed in all_seeds:
                v = 0
                name = f"{model_name}+target={config['target']}+seed={seed}"
                while os.path.exists(os.path.join(base, f"{name}-{v}")):
                    v += 1
                output_dir = os.path.join(base, f"{name}-v{v}")

                os.makedirs(output_dir, exist_ok=True)
                config_with_seed = copy.deepcopy(config)
                config_with_seed["seed"] = seed
                hparams_path = os.path.join(output_dir, "hparams.yaml")
                yaml.safe_dump(config_with_seed, open(hparams_path, "w"))
                 
                function = submitit.helpers.CommandFunction([
                    'python3', 'main.py',
                    '--config_default', hparams_path,
                    '--output_dir', output_dir,
                ])
                print(' '.join(function.command))
                job = executor.submit(function)
                jobs.append(job)
