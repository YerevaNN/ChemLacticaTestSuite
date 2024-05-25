import yaml
import os
from pathlib import Path
from main.optimizer import top_auc
from tqdm import tqdm
import numpy as np
import multiprocessing
from functools import partial
import argparse


def process_tune_result(dir, hparams_to_log):
    if not os.path.exists(os.path.join(dir, "hparams.yaml")):
        return None
    hparams = yaml.safe_load(open(os.path.join(dir, "hparams.yaml")))
    config_repr = ""
    for k, v in hparams_to_log.items():
        if '+' in k:
            a, b = k.split('+')
            config_repr += f"{v}-{hparams[a][b]}-"
        else:
            config_repr += f"{v}-{hparams[k]}-"

    result_files = list(Path(dir).glob("*.yaml"))
    result_files.remove(Path(os.path.join(dir, "hparams.yaml")))
    result_files_with_names = {}
    for file in result_files:
        task_name = file.stem[:-3].removeprefix('results_chemlactica_')
        if not result_files_with_names.get(task_name):
            result_files_with_names[task_name] = []
        result_files_with_names[task_name].append(file)

    result_pairs = []
    for task_name, files in result_files_with_names.items():
        auc_top10s = []
        avg_top10s = []
        for file in files:
            # print(file)
            try:
                result_yaml = yaml.safe_load(open(file))
                if len(result_yaml) < 10000:
                    print(f"WARNING: {dir} {file.name} only has {len(result_yaml)} molecules")
                    continue
                auc_top10 = top_auc(result_yaml, top_n=10, finish=True, freq_log=100, max_oracle_calls=10000)
                scores = [item[1][0] for item in list(result_yaml.items())]
                avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
                avg_top10s.append(avg_top10)
                auc_top10s.append(auc_top10)
            except Exception as e:
                print(e)
        if len(auc_top10s) != 0:
            result_pairs.append((task_name, np.mean(auc_top10s), np.std(auc_top10s), np.mean(avg_top10s), np.std(avg_top10s)))
    result_pairs.sort(key=lambda x:x[0])
    if len(result_pairs) != 0:
        return config_repr, result_pairs
    return None


def process_tune_results(root_dir):
    hparams_to_log = {
        "num_mols": "C",
        "num_similars": "S",
        "rej_sample_config+max_learning_rate": "lr"
    }
    result_pairs = []
    output_lines = []
    all_dirs = Path(root_dir).iterdir()
    with multiprocessing.Pool(processes=8) as pol:
        for res in pol.map(partial(process_tune_result, hparams_to_log=hparams_to_log), all_dirs):
            if res:
                config_repr, result_pairs = res
                auc_sum = 0
                avg_sum = 0
                line = config_repr + ": "
                for task_name, mean, std, avg_mean, avg_std in result_pairs:
                    line += f"{task_name}: AUC-{mean:.3f} \pm {std:.3f}, AVG-{avg_mean:.3f} \pm {avg_std:.3f}, "
                    auc_sum += mean
                    avg_sum += avg_mean
                
                line += f"sum: AUC-{auc_sum:.3f}, AVG-{avg_sum:.3f}"
                output_lines.append(line)
                print("constructed for", config_repr)

    output_lines.sort()
    with open(f"{root_dir}/tune-results.log", "w") as _file:
        for line in output_lines:
            _file.write(line + "\n")
        # for res in pol.map(partial(process_tune_result, hparams_to_log=hparams_to_log), all_dirs):
        #     if res:
        #         result_pairs.extend(res)
    # result_pairs.sort(key=lambda x:x[0])
    # for task_name, mean, std, avg_mean, avg_std in result_pairs:
    #     print(f"{task_name} - AUC {mean:.3f} \pm {std:.3f}, AVG {avg_mean:.3f} \pm {avg_std:.3f}")


def inner_proc(e):
    task_name, files = e
    auc_top10s = []
    avg_top10s = []
    diversity_scores = []
    for file in files:
        # print(file)
        try:
            result_yaml = yaml.safe_load(open(file))
            if len(result_yaml) < 10000:
                print(f"WARNING: {file.name} only has {len(result_yaml)} molecules")
                continue
            auc_top10 = top_auc(result_yaml, top_n=10, finish=True, freq_log=100, max_oracle_calls=10000)
            scores = [item[1][0] for item in list(result_yaml.items())]
            avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
            avg_top10s.append(avg_top10)
            auc_top10s.append(auc_top10)
            diversity_scores.append
        except Exception as e:
            print(e)
    if len(auc_top10s) != 0:
        return (task_name, np.mean(auc_top10s), np.std(auc_top10s), np.mean(avg_top10s), np.std(avg_top10s))
    return None


def process_results(dir):
    result_files = list(Path(dir).glob("*.yaml"))
    result_files.remove(Path(os.path.join(dir, "hparams.yaml")))
    result_files_with_names = {}
    for file in result_files:
        task_name = file.stem[:-3].removeprefix('results_chemlactica_')
        if not result_files_with_names.get(task_name):
            result_files_with_names[task_name] = []
        result_files_with_names[task_name].append(file)

    result_pairs = []
    with multiprocessing.Pool(processes=8) as pol:
        # print(f'count {len(result_files_with_names)}')
        progress_bar = tqdm(total=len(result_files_with_names))
        for res in pol.imap(inner_proc, result_files_with_names.items()):
            if res:
                result_pairs.append(res)
            progress_bar.update(1)
    result_pairs.sort(key=lambda x:x[0])
    auc_sum = 0
    avg_sum = 0
    for task_name, mean, std, avg_mean, avg_std in result_pairs:
        print(f"{task_name} - AUC {mean:.3f} \pm {std:.3f}, AVG {avg_mean:.3f} \pm {avg_std:.3f}")
        auc_sum += mean
        avg_sum += avg_mean
    
    print(f"AUC {auc_sum:.3f}, AVG {avg_sum:.3f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=False, default='default')
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    if args.type == 'tune':
        process_tune_results(args.path)
    else:
        process_results(args.path)