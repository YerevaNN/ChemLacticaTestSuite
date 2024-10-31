import submitit
import itertools as it
import yaml
import datetime
import os
import copy


def create_hparam_configs(config_file_path):
    config_tune = yaml.safe_load(open("chemlactica/hparams_tune.yaml"))
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


if __name__ == "__main__":

    config_file_path = "chemlactica/chemlactica_125m_hparams.yaml"
    # config_file_path = "chemlactica/chemlactica_1.3b_hparams.yaml"
    # hparam_configs = create_hparam_configs(config_file_path)
    infer_config = [yaml.safe_load(open(config_file_path))]
    # hparam_configs = create_hparam_configs(config_file_path)
    model_name = "-".join(config_file_path.split("/")[-1].split("_")[:2])

    executor = submitit.AutoExecutor(folder="/auto/home/tigranfahradyan/slurm_jobs/RetMol/job_%j")
    executor.update_parameters(
        name="chemlactica-retmol-qed+sim",
        timeout_min=12 * 60, gpus_per_node=1, nodes=1,
        mem_gb=50, slurm_array_parallelism=10, cpus_per_task=2
    )
    jobs = []
    with executor.batch():
        for config in infer_config:
            formatted_date_time = datetime.datetime.now().strftime("%Y-%m-%d")
            base = f"chemlactica/results/{formatted_date_time}"
            os.makedirs(base, exist_ok=True)
            v = 0
            name = model_name + "-" + "+".join(config["strategy"])
            while os.path.exists(os.path.join(base, f"{name}-{v}")):
                v += 1
            output_dir = os.path.join(base, f"{name}-{v}-qed+sim")
            os.makedirs(output_dir, exist_ok=True)
            yaml.safe_dump(config, open(os.path.join(output_dir, "hparams.yaml"), "w"))
            function = submitit.helpers.CommandFunction([
                'python3', 'chemlactica/run_qed.py',
                '--run_name', 'qed+sim',
                '--qed_csv_path', 'data/zinc/inference_inputs/qed_test_w_vals.csv',
                '--config_default', os.path.join(output_dir, "hparams.yaml"),
                '--output_dir', output_dir
            ])
            print(' '.join(function.command))
            # job = executor.submit(function)
            # jobs.append(job)