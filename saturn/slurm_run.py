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
    model_name = "chemma_2b"
    # model_name = "chemlactica_1.3b"
    # model_name = "chemlactica_125m"

    config_file_path = f"{model_name}_hparams.yaml"
    default_config = [yaml.safe_load(open(config_file_path, "r"))]

    executor = submitit.AutoExecutor(folder="~/slurm_jobs/saturn/job_%j")
    executor.update_parameters(
        name="chemlactica-saturn", timeout_min=5 * 60,
        gpus_per_node=1,
        nodes=1, mem_gb=80, cpus_per_task=4,
        slurm_array_parallelism=16,
        additional_parameters={
            "partition": "h100",
            # "gres": "shard:39"
        }
    )

    jobs = []
    with executor.batch():
        for config in default_config:
            formatted_date_time = datetime.datetime.now().strftime("%Y-%m-%d")
            base = os.path.join("results", formatted_date_time)
            os.makedirs(base, exist_ok=True)
            name = f"{model_name}"
            v = 0
            while os.path.exists(os.path.join(base, f"{name}-{v}")):
                v += 1
            output_dir = os.path.join(base, f"{name}-{v}")
            os.makedirs(output_dir, exist_ok=True)

            hparams_path = os.path.join(output_dir, f"hparams.yaml")
            yaml.safe_dump(config, open(hparams_path, "w"))
            for target in target_proteins:
                for seed in all_seeds:
                    function = submitit.helpers.CommandFunction([
                        'python3', 'main.py',
                        '--config_default', hparams_path,
                        '--output_dir', output_dir,
                        '--seed', str(seed),
                        '--target', target,
                        '--budget', str(3000)
                    ])
                    print(' '.join(function.command))
                    job = executor.submit(function)
                    jobs.append(job)
