import argparse
import math
import seaborn as sns
import matplotlib.pyplot as plt
import json



def main():
    # with open(args.config, 'r') as f:
    #     config = yaml.safe_load(f)
    plot_dict = {}

    for file_path in ["./results/Gemma2b_best_qed_perps/Gemma2b_best_qed_perps.log","./results/gal125m_best_qed_perps/gal125m_best_qed_perps.log"]:
        norm_perps = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line_data = json.loads(line)
            normalized_perp = math.log(line_data["ppl"]) * line_data["num_tokens"]
            norm_perps.append(normalized_perp)
        plot_dict[file_path] = norm_perps

    sns.set(style="whitegrid")  
    plt.figure(figsize=(10, 6))  

    for key_file_path, norm_perps in plot_dict.items():
        sns.histplot(norm_perps, bins=20, label=key_file_path, kde=True, stat="density")
    plt.legend()

    # Add labels and title
    plt.xlabel('Normalized Perplexity')
    plt.ylabel('Density')
    plt.title('Histogram of perplexities for 100 correct QED->SMILES samples')
    plt.savefig('perplexities_hist.png')




if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True)
    # args = parser.parse_args()
    main()
