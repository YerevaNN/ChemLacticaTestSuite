import pprint
import argparse
import pandas as pd

# The median docking scores for each target protein
# The numbers are taken from https://github.com/SeulLee05/MOOD/blob/main/evaluate.py used by MOOD https://arxiv.org/pdf/2206.07632
# which is referenced in the SATURN paper
MEDIAN_DOCKING_SCORES_PER_TARGET = {
    "fa7": 8.5,
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
    
    qed_threshold = 0.7 if strict else 0.5
    sa_threshold = 3 if strict else 5
    
    filtered = df[df['vina'] > -median_docking_score] 
    filtered = filtered[filtered['qed'] > qed_threshold]
    filtered = filtered[filtered['sa'] < sa_threshold]

    # Todo include also checking of tanimoto similarity score against training set
    if novel:
        pass

    return len(filtered) / len(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, help='The target protein that was used to dock against durint optimization.')
    parser.add_argument('--log_file', type=str, help='Path to the log file containing the optimization run logs.')
    args = parser.parse_args()

    with open(args.log_file, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            if line.startswith("generated smiles:"):
                parts = line.split(",")
                row = {}
                for part in parts:
                    part = part.strip()
                    col, value = part.split(":") 
                    value = value.strip()
                    try:
                        value = float(value)
                    except:
                        pass

                    row[col.strip()] = value
                data.append(row)

    df = pd.DataFrame(data)
    
    pprint.pprint({
        "Hit Ratio": calculate_hit_ratio(df, args.target),
        "Strict Hit Ratio": calculate_hit_ratio(df, args.target, strict=True),
        "Novel Hit Ratio": calculate_hit_ratio(df, args.target, novel=True),
        "Strict Novel Hit Ratio": calculate_hit_ratio(df, args.target, strict=True, novel=True)
    })