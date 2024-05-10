import matplotlib.pyplot as plt
import os
import argparse
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import importlib
import warnings

from chemlactica import mol_opt
importlib.reload(mol_opt)
from chemlactica.mol_opt.optimization import optimize
from chemlactica.mol_opt.utils import MoleculeEntry, tanimoto_dist_func

def get_line(line, s, e='\n'):
    if line.find(s) != -1 and line.find(e) != -1:
        return line[line.find(s) + len(s): line.find(e)]
    return None

def read_log_file(path, count=None):
    with open(path, "r") as _f:
        all_lines = _f.readlines()
    optimization_timeline = []
    mode = 'gen_mol'
    if not count:
        count = len(all_lines)
    for line in tqdm(all_lines[:count]):
        if 'generated smiles: ' in line:
            smiles = get_line(line, 'generated smiles: ', ', score: ')
            smiles_score = float(get_line(line, ', score: '))
            try:
                optimization_timeline.append({"smiles_gen": MoleculeEntry(smiles, smiles_score)})
            except:
                pass
            mode = 'gen_mol'

        if mode == 'pool':
            # print(line)
            pool_smiles = get_line(line, 'smiles: ', ', score: ')
            pool_smiles_score = float(get_line(line, ', score: '))
            try:
                optimization_timeline[-1]["pool"].append(MoleculeEntry(pool_smiles, pool_smiles_score))
            except:
                pass
        if 'Molecule pool' in line:
            mode = 'pool'
            optimization_timeline.append({"pool": []})
        if mode == 'tr_data':
            tr_smiles = get_line(line, 'smiles: ', ', score: ')
            tr_smiles_score = float(get_line(line, ', score: '))
            try:
                optimization_timeline[-1]["tr_data"].append(MoleculeEntry(tr_smiles, tr_smiles_score))
            except:
                pass
        if 'Training entries' in line:
            mode = 'tr_data'
            optimization_timeline.append({"tr_data": []})
        if 'Dump ' in line:
            optimization_timeline.append({"pool_dump": 25})
    return optimization_timeline

def plot_optim_timeline(optimization_timeline, title, y_top=1.0):
    # optimization_timeline = optimization_timeline[:20000]
    print(len(optimization_timeline))
    sns.set_style("whitegrid")
    
    scores = []
    scores_time = []
    pool_mols = []
    pool_time = []
    tr_data_mols = []
    tr_data_time = []
    pool_dump_time = []
    for i, entry in enumerate(optimization_timeline):
        if entry.get('smiles_gen'):
            scores.append(entry['smiles_gen'].score)
            scores_time.append(i)
        if entry.get('pool'):
            pool_mols.append(entry['pool'])
            pool_time.append(i)
        if entry.get('tr_data'):
            tr_data_mols.append(entry['tr_data'])
            tr_data_time.append(i)
        if entry.get('pool_dump'):
            pool_dump_time.append(i)

    # draw generated molecules
    scores = np.array(scores)
    scores_time = np.array(scores_time)
    rand_ind = np.random.permutation(len(scores))
    scores = scores[rand_ind[:2000]]
    scores_time = scores_time[rand_ind[:2000]]
    sns.scatterplot(data={'scores': scores, 'time': scores_time}, x='time', y='scores', markers='', s=10, label='generated mols')
    # plt.legend('generated mols')

    # draw pool molecules
    pool_time = np.array(pool_time)
    label = 'pool mols'
    for e in zip(*pool_mols):
        pool_scores_ = np.array([m.score for m in e])
        rand_ind = np.random.permutation(len(pool_scores_))
        sns.scatterplot(data={'scores': pool_scores_[rand_ind[:50]], 'time': pool_time[rand_ind[:50]]}, x='time', y='scores', s=10, color='pink', label=label)
        label = None

    # draw training molecules
    tr_data_time = np.array(tr_data_time)
    label = 'train mols'
    for e in zip(*tr_data_mols):
        tr_data_scores_ = np.array([m.score for m in e])
        # rand_ind = np.random.permutation(len(tr_data_scores_))
        sns.scatterplot(data={'scores': tr_data_scores_, 'time': tr_data_time}, x='time', y='scores', s=10, color='red', label=label)
        label = None

    for t in pool_dump_time:
        sns.lineplot(x=np.array([t, t]), y=np.array([0, 1]), color='red')
    
    plt.title(title)
    plt.ylim(0.0, y_top)
    plt.savefig(f'{title}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file_path", type=str, required=True, help="Path to log file to visualize")
    parser.add_argument("--title", type=str, required=True, help="Path to output plot file")
    args = parser.parse_args()
    full_log_file_path = os.path.abspath(args.log_file_path)

    optimization_timeline = read_log_file(full_log_file_path)
    plot_optim_timeline(optimization_timeline, title=args.title)
