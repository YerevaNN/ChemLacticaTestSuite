from typing import List
import random
import math
import sys
import os
from pathlib import Path
import random
import torch
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.Chem.QED import qed

from chemlactica.mol_opt.utils import tanimoto_dist_func
sys.path.append("./inference")
from utils_inference import sascorer

# Disable RDKit logs
RDLogger.DisableLog('rdApp.*')
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class LeadOptimizationQEDOracle:

    def __init__(self, lead_entry, max_oracle_calls: int):
        self.lead_entry = lead_entry
        self.max_oracle_calls = max_oracle_calls
        self.found_opt_mol = False
        self.mol_buffer = {}
        self.takes_entry = True
        self.qed_bound = 0.9
        self.sim_bound = 0.4

    def __call__(self, molecules):
        oracle_scores = []
        for molecule in molecules:
            if self.mol_buffer.get(molecule.smiles):
                oracle_scores.append(sum(self.mol_buffer[molecule.smiles][0]))
            else:
                try:
                    qed_ = qed(molecule.mol)
                    sim_to_lead = tanimoto_dist_func(molecule.fingerprint, self.lead_entry.fingerprint)
                    if qed_ >= self.qed_bound and sim_to_lead >= self.sim_bound:
                        self.found_opt_mol = True

                    # if qed_ >= self.qed_bound:
                    #     qed_mod = 1
                    # else:
                    #     qed_mod = qed_ ** 2 * (0.5 / self.qed_bound ** 2)
                    # if sim_to_lead >= self.sim_bound:
                    #     sim_to_lead_mod = 1
                    # else:
                    #     sim_to_lead_mod = sim_to_lead ** 2 * (0.5 / self.sim_bound ** 2)
                    qed_mod = qed_
                    sim_to_lead_mod = (0.9 / self.sim_bound) * sim_to_lead
                    if sim_to_lead > self.sim_bound:
                        sim_to_lead_mod = (1-0.9) / (1-self.sim_bound) * sim_to_lead + 1 - (1-0.9) / (1-self.sim_bound)
                except Exception as e:
                    print(e)
                    qed_mod = 0
                    sim_to_lead_mod = 0
                oracle_score = qed_mod + sim_to_lead_mod
                self.mol_buffer[molecule.smiles] = [(qed_mod, sim_to_lead_mod), len(self.mol_buffer) + 1]
                if len(self.mol_buffer) % 100 == 0:
                    self.log_intermediate()
                oracle_scores.append(oracle_score)
        return oracle_scores
    
    def __len__(self):
        return len(self.mol_buffer)
    
    def log_intermediate(self):
        scores = [v[0][0] + v[0][1] for v in self.mol_buffer.values()]
        qed_values = [v[0][0] for v in self.mol_buffer.values()]
        sim_to_lead_values = [v[0][1] for v in self.mol_buffer.values()]
        scores_sorted = sorted(scores, reverse=True)[:100]
        qed_sorted = sorted(qed_values, reverse=True)[:100]
        sim_to_lead_sorted = sorted(sim_to_lead_values, reverse=True)[:100]
        n_calls = len(self.mol_buffer)

        score_avg_top1 = np.max(scores_sorted)
        score_avg_top10 = np.mean(scores_sorted[:10])
        score_avg_top100 = np.mean(scores_sorted)

        qed_avg_top1 = np.max(qed_sorted)
        qed_avg_top10 = np.mean(qed_sorted[:10])
        qed_avg_top100 = np.mean(qed_sorted)

        sim_avg_top1 = np.max(sim_to_lead_sorted)
        sim_avg_top10 = np.mean(sim_to_lead_sorted)
        sim_avg_top100 = np.mean(sim_to_lead_sorted)
        
        print(f'{n_calls}/{self.max_oracle_calls} | '
                f'score_avg_top1: {score_avg_top1:.3f} | '
                f'score_avg_top10: {score_avg_top10:.3f} | '
                f'score_avg_top100: {score_avg_top100:.3f}')

        print(f'qed_avg_top1: {qed_avg_top1:.3f} | '
                f'qed_avg_top10: {qed_avg_top10:.3f} | '
                f'qed_avg_top100: {qed_avg_top100:.3f}')
        
        print(f'sim_avg_top1: {sim_avg_top1:.3f} | '
                f'sim_avg_top10: {sim_avg_top10:.3f} | '
                f'sim_avg_top100: {sim_avg_top100:.3f}')

    @property
    def budget(self):
        return self.max_oracle_calls

    @property
    def finish(self):
        return self.found_opt_mol or len(self.mol_buffer) >= self.max_oracle_calls


class LeadOptimizationPlogPOracle:

    def __init__(self, lead_entry, max_oracle_calls: int, sim_bound: int):
        self.lead_entry = lead_entry
        self.max_oracle_calls = max_oracle_calls
        self.mol_buffer = {}
        self.takes_entry = True
        self.sim_bound = sim_bound

    def __call__(self, molecules):
        oracle_scores = []
        for molecule in molecules:
            if self.mol_buffer.get(molecule.smiles):
                oracle_scores.append(self.mol_buffer[molecule.smiles][0])
            else:
                try:
                    clogp = Descriptors.MolLogP(molecule.mol) # assume clopg is in [-5, 30]
                    sas = sascorer.calculateScore(molecule.mol) # sas is in [1, 10]
                    plogp = clogp - sas # assume plogp is in [-15, 31]

                    if plogp <= -15:
                        plogp_mod = 0
                    elif plogp >= 31:
                        plogp_mod = 1
                    elif plogp >= 0:
                        plogp_mod = 1 / (1 + math.exp(-plogp / 30))
                    else:
                        plogp_mod = 1 / (1 + math.exp(-plogp))

                    sim_to_lead = tanimoto_dist_func(molecule.fingerprint, self.lead_entry.fingerprint)
                    if sim_to_lead <= self.sim_bound:
                        sim_to_lead_mod = (0.9 / self.sim_bound) * sim_to_lead
                    if sim_to_lead > self.sim_bound:
                        sim_to_lead_mod = (1-0.9) / (1-self.sim_bound) * sim_to_lead + 1 - (1-0.9) / (1-self.sim_bound)
                except Exception as e:
                    print(e)
                    plogp_mod = 0
                    sim_to_lead_mod = 0
                oracle_score = plogp_mod + sim_to_lead_mod
                self.mol_buffer[molecule.smiles] = [oracle_score, plogp, sim_to_lead, len(self.mol_buffer) + 1]
                if len(self.mol_buffer) % 100 == 0:
                    self.log_intermediate()
                oracle_scores.append(oracle_score)
        return oracle_scores
    
    def __len__(self):
        return len(self.mol_buffer)
    
    def log_intermediate(self):
        scores = [v[0] for v in self.mol_buffer.values()]
        plogp_values = [v[1] for v in self.mol_buffer.values() if v[2] >= self.sim_bound]
        scores_sorted = sorted(scores, reverse=True)[:100]
        plogp_sorted = sorted(plogp_values, reverse=True)[:100]
        n_calls = len(self.mol_buffer)

        score_avg_top1 = np.max(scores_sorted)
        score_avg_top10 = np.mean(scores_sorted[:10])
        score_avg_top100 = np.mean(scores_sorted)

        plogp_avg_top1 = np.max(plogp_sorted)
        plogp_avg_top10 = np.mean(plogp_sorted[:10])
        plogp_avg_top100 = np.mean(plogp_sorted)

        print(f'{n_calls}/{self.max_oracle_calls} | '
                f'score_avg_top1: {score_avg_top1:.3f} | '
                f'score_avg_top10: {score_avg_top10:.3f} | '
                f'score_avg_top100: {score_avg_top100:.3f}')

        print(f'plogp_avg_top1: {plogp_avg_top1:.3f} | '
                f'plogp_avg_top10: {plogp_avg_top10:.3f} | '
                f'plogp_avg_top100: {plogp_avg_top100:.3f}')
        
    @property
    def budget(self):
        return self.max_oracle_calls

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls
