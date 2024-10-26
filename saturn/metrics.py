import pprint
import argparse
import pandas as pd

# The median docking scores for each target protein
# The numbers are taken from https://github.com/SeulLee05/MOOD/blob/main/evaluate.py used by MOOD https://arxiv.org/pdf/2206.07632
# which is referenced in the SATURN paper
MEDIAN_DOCKING_SCORES_PER_TARGET = {
    "fa7": 8.5, # < 8.5
    "parp1": 10.0,
    "5ht1b": 8.7845,
    "jak2": 9.1,
    "braf": 10.3,
}


def calculate_hit_ratio(df: pd.DataFrame, target: str, strict: bool = False, novel: bool = False) -> float:
    try:
        median_docking_score = MEDIAN_DOCKING_SCORES_PER_TARGET[target]
    except KeyError as e:
        raise ValueError(f"Target protein {target} is not supported.") from e
    
    qed_threshold = 0.7 if strict else 0.5 # qed > 0.7
    sa_threshold = 3 if strict else 5 # sas < 3
    
    filtered = df[df['vina'] < median_docking_score]
    filtered = filtered[filtered['qed'] > qed_threshold]
    filtered = filtered[filtered['sa'] < sa_threshold]

    # Todo include also checking of tanimoto similarity score against training set
    if novel:
        pass

    return len(filtered) / len(df)


def calculate_metrics_from_log_file(log_file: str):
    file_name = log_file.split("/")[-1]
    target = file_name.split("+")[1]

    with open(log_file, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            if line.startswith("generated smiles:"):
                parts = line.split(",")
                row = {}
                for part in parts:
                    part = part.strip()
                    col, value = part[:part.find(":")], part[part.find(":") + 1:]
                    # col, value = part.split(":")
                    value = value.strip()
                    try:
                        value = float(value)
                    except:
                        pass

                    row[col.strip()] = value
                data.append(row)

    # assert len(data) >= 3000
    if len(data) < 3000:
        print(f"WARNING {log_file} has {len(data)} entries.")
        return None
    data = data[:3000]
    df = pd.DataFrame(data)
    
    return {
        "Hit Ratio": calculate_hit_ratio(df, target),
        "Strict Hit Ratio": calculate_hit_ratio(df, target, strict=True),
        "Novel Hit Ratio": calculate_hit_ratio(df, target, novel=True),
        "Strict Novel Hit Ratio": calculate_hit_ratio(df, target, strict=True, novel=True)
    } 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, help="The target protein to calculate the metrics for.")
    parser.add_argument('--logs_dir', type=str, help='Path to the log file containing the optimization run logs.')
    args = parser.parse_args()

    # List all files in the logs directory that contain the target protein name in their file name
    import os
    metrics_df = []

    paths = [p for p in os.listdir(args.logs_dir) if args.target in p and p.endswith(".log")]
    print(len(paths))
    if len(paths) != 10:
        print(f"WARNING the number of log files is {len(paths)}")
    # assert len(paths) == 10

    for log_file in paths:
        metrics = calculate_metrics_from_log_file(os.path.join(args.logs_dir, log_file))
        if metrics:
            metrics_df.append(metrics)

    metrics_df = pd.DataFrame(metrics_df) * 100
    print(f"Metrics for target protein {args.target}\n")
    print(f"Hit Ratio: {metrics_df['Hit Ratio'].mean().round(3)} ± {metrics_df['Hit Ratio'].std().round(3)}")
    print(f"Strict Hit Ratio: {metrics_df['Strict Hit Ratio'].mean().round(3)} ± {metrics_df['Strict Hit Ratio'].std().round(3)}")
    # print(metrics_df.describe().round(2))
