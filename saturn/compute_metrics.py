import argparse
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm
import pandas as pd
import os


def get_line(line, s, e='\n'):
    if line.find(s) != -1 and line.find(e) != -1:
        return line[line.find(s) + len(s): line.find(e)]
    return None


def calc_generative_yield(scores, th_score):
    return np.sum(scores >= th_score)


def calc_oracle_burden(scores, th_score, count):
    inds = np.where(scores >= th_score)[0] - 1
    if len(inds) < count:
        return None
    return inds[count - 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dir", type=str, required=True)
    parser.add_argument("--max_oracle_calls", type=int, required=True)
    parser.add_argument("--type", type=str, default="exp", required=False)

    args = parser.parse_args()

    metric_names = [
        "gen_yield_0.7",
        "gen_yield_0.8",
        "oracle_burden_0.8_1",
        "oracle_burden_0.8_10",
        "oracle_burden_0.8_100",
        "oracle_burden_0.7_1",
        "oracle_burden_0.7_10",
        "oracle_burden_0.7_100"
    ]

    if args.type == "tune":
        print("Computing metrics tuning hparams.")
        track_params = {
            "checkpoint_path": "model",
            "pool_size": "P",
            "num_gens_per_iter": "N",
            "num_similars": "S",
            "generation_temperature": "temp",
            "rej_sample_config.train_tol_level": "tol_level",
            "rej_sample_config.max_learning_rate": "lr"
        }
        data = []
        for d in tqdm(Path(args.path_to_dir).iterdir()):
            if not d.is_dir():
                continue
            hparams = yaml.safe_load(open(os.path.join(str(d), "hparams.yaml"), "r"))
            # if hparams["generation_temperature"][1] > hparams["generation_temperature"][0]:
            #     continue
            config_name = str(d) + ", "
            for param in track_params.keys():
                if "." in param:
                    a, b = param.split('.')
                    value = hparams[a][b]
                else:
                    value = hparams[param]
                config_name += f"{track_params[param]}: {value}, "
            
            config_name = config_name[:-2]
            metrics = {n: [] for n in metric_names}
            for log_file in d.iterdir():
                if not str(log_file).endswith(".log"):
                    continue
                scores = []
                with open(log_file, "r") as _f:
                    for l in _f.readlines():
                        # print(l.find('aggregate score:'), l.find('\n'))
                        score = get_line(l, "aggregate score: ", ", \n")
                        if score:
                            scores.append(float(score))
                if len(scores) < args.max_oracle_calls:
                    print(f"Warning: {d} does not have enough generated molecules ({len(scores)})")
                    # for n in metric_names:
                    #     metrics[n].append(None)
                    continue
                scores = np.array(scores[:args.max_oracle_calls])
                metrics["gen_yield_0.7"].append(calc_generative_yield(scores, 0.7))
                metrics["gen_yield_0.8"].append(calc_generative_yield(scores, 0.8))
                metrics["oracle_burden_0.8_1"].append(calc_oracle_burden(scores, 0.8, 1))
                metrics["oracle_burden_0.8_10"].append(calc_oracle_burden(scores, 0.8, 10))
                metrics["oracle_burden_0.8_100"].append(calc_oracle_burden(scores, 0.8, 100))
                metrics["oracle_burden_0.7_1"].append(calc_oracle_burden(scores, 0.7, 1))
                metrics["oracle_burden_0.7_10"].append(calc_oracle_burden(scores, 0.7, 10))
                metrics["oracle_burden_0.7_100"].append(calc_oracle_burden(scores, 0.7, 100))

            mean_std_metrics = []
            for key in metric_names:
                try:
                    # assert len(metrics[key]) >= 4
                    metric_str = f"{np.mean(metrics[key]).round(4)} ± {np.std(metrics[key]).round(4)}"
                    metric_str += f" ({len(metrics[key])})"
                except Exception as e:
                    print(e)
                    metric_str = "Failed"
                mean_std_metrics.append(metric_str)

            data.append([config_name] + mean_std_metrics)
            print(f"finished {str(d)}")

        ordered_data = sorted(data, key=lambda x: x[0])
        # print(ordered_data)

        res_df = pd.DataFrame(data=ordered_data, columns=["config"] + metric_names)
        res_df.to_csv(os.path.join(args.path_to_dir, f"tune-results.csv"))
    else:
        d = Path(args.path_to_dir)
        hparams = yaml.safe_load(open(os.path.join(str(d), "hparams.yaml"), "r"))
        config_name = str(d)

        for target in ["parp1", "fa7", "5ht1b", "jak2", "braf"]:
            data = []
            metrics = {n: [] for n in metric_names}
            for log_file in d.iterdir():
                if not log_file.name.endswith(".log") or not target in log_file.name:
                    continue
                scores = []
                with open(log_file, "r") as _f:
                    for l in _f.readlines():
                        # print(l.find('aggregate score:'), l.find('\n'))
                        score = get_line(l, "aggregate score: ", ", vina")
                        if score:
                            scores.append(float(score))
                if len(scores) < args.max_oracle_calls:
                    print(f"Warning: {d} does not have enough generated molecules ({len(scores)})")
                    # for n in metric_names:
                    #     metrics[n].append(None)
                    continue
                scores = np.array(scores[:args.max_oracle_calls])
                print(scores)
                metrics["gen_yield_0.7"].append(calc_generative_yield(scores, 0.7))
                metrics["gen_yield_0.8"].append(calc_generative_yield(scores, 0.8))
                metrics["oracle_burden_0.8_1"].append(calc_oracle_burden(scores, 0.8, 1))
                metrics["oracle_burden_0.8_10"].append(calc_oracle_burden(scores, 0.8, 10))
                metrics["oracle_burden_0.8_100"].append(calc_oracle_burden(scores, 0.8, 100))
                metrics["oracle_burden_0.7_1"].append(calc_oracle_burden(scores, 0.7, 1))
                metrics["oracle_burden_0.7_10"].append(calc_oracle_burden(scores, 0.7, 10))
                metrics["oracle_burden_0.7_100"].append(calc_oracle_burden(scores, 0.7, 100))

            mean_std_metrics = []
            for key in metric_names:
                try:
                    # assert len(metrics[key]) >= 4
                    metric_str = f"{np.mean(metrics[key]).round(4)} ± {np.std(metrics[key]).round(4)}"
                    metric_str += f" ({len(metrics[key])})"
                except Exception as e:
                    print(e)
                    metric_str = "Failed"
                mean_std_metrics.append(metric_str)

        data.append([config_name] + mean_std_metrics)

    ordered_data = sorted(data, key=lambda x: x[0])
    # print(ordered_data)

    res_df = pd.DataFrame(data=ordered_data, columns=["config"] + metric_names)
    res_df.to_csv(os.path.join(args.path_to_dir, f"results_gen_yield_oracle_burden.csv"))
        