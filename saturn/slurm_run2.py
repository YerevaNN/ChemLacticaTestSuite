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

    config_file_path = "chemlactica_125m_hparams.yaml"
    # config_file_path = "chemlactica_1.3b_hparams.yaml"
    # config_file_path = "chemma_2b_hparams.yaml"
    default_config = yaml.safe_load(open(config_file_path))
    infer_configs = [default_config]
    model_name = "-".join(config_file_path.split("/")[-1].split("_")[:2])

    executor = submitit.AutoExecutor(folder="~/slurm_jobs/saturn/job_%j")
    executor.update_parameters(
        name="chemlactica-against-saturn", timeout_min=int(4 * 60),
        gpus_per_node=1,
        nodes=1, mem_gb=30, cpus_per_task=10,
        slurm_array_parallelism=10
    )
    
    jobs = []
    with executor.batch():
        for config in infer_configs:
            formatted_date_time = datetime.datetime.now().strftime("%Y-%m-%d")
            base = f"results/{formatted_date_time}"
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
                        'python3', 'run_exp.py',
                        '--config_default', hparams_path,
                        '--output_dir', output_dir,
                        '--target', target,
                        '--seed', str(seed)
                    ])
                    print(' '.join(function.command))
                    job = executor.submit(function)
                    jobs.append(job)
