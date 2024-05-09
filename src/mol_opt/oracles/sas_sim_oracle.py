import sys
import numpy as np
from chemlactica.mol_opt.utils import MoleculeEntry, tanimoto_dist_func
from .oracle_base import BaseOptimizationOracle
from rdkit.Contrib.SA_Score import sascorer
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


class InheritedLeadOptimizationSASOracle(BaseOptimizationOracle):
    def __init__(self, sas_constraint_value: float, max_oracle_calls: int):
        super().__init__(max_oracle_calls,takes_entry=True)
        self.lead_entry = None
        self.sas_constraint_value = sas_constraint_value
        self.best_molecule_entry = None
        self.success_rate = 0
        self.num_lead_molecules_seen = 0
        self.num_molecules_optimized = 0
        self.max_sim_mean = 0
        self.max_sim_sum = 0
        self.num_optimized = 0
    
    def set_lead_entry(self, lead_entry):
        self.lead_entry = lead_entry
        self.num_lead_molecules_seen +=1

    def _store_custom_data(self):
        self.max_sim_sum += self.best_molecule_entry.add_props["sim_to_lead"] if self.best_molecule_entry is not None else 0
        self.num_optimized += 1 if self.best_molecule_entry is not None and (self.best_molecule_entry.add_props["sim_to_lead"] >= 0.4) else 0

    def _clear_custom_data(self):
        self.best_molecule_entry = None
        self.lead_entry = None

    def _log_intermediate(self):
        all_values = [v[0] for v in self.mol_buffer.values()]
        scores = sorted(all_values, reverse=True)[:100]
        n_calls = len(self.mol_buffer)
        
        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        
        print(f'{n_calls}/{self.max_oracle_calls} | '
                f'avg_top1: {avg_top1:.3f} | '
                f'avg_top10: {avg_top10:.3f} | '
                f'avg_top100: {avg_top100:.3f}')

        # try:
        print({
            "avg_top1": avg_top1, 
            "avg_top10": avg_top10, 
            "avg_top100": avg_top100,
            "n_oracle": n_calls,
        })

    def _calculate_score(self, mol_entry):
        sas = sascorer.calculateScore(mol_entry.mol)
        if sas > self.sas_constraint_value:
            oracle_score = 0
        else:
            sim_to_lead = tanimoto_dist_func(mol_entry.fingerprint, self.lead_entry.fingerprint)
            oracle_score = sim_to_lead
            if not self.best_molecule_entry or self.best_molecule_entry.add_props["sim_to_lead"] < sim_to_lead:
                self.best_molecule_entry = mol_entry
                self.best_molecule_entry.add_props["sim_to_lead"] = sim_to_lead
                self.best_molecule_entry.add_props["sas"] = sas
        return oracle_score

    def _calculate_metrics(self):

        self.metrics_dict["success_rate"] = self.num_optimized / self.num_lead_molecules_seen
        self.metrics_dict["max_sim_sum"] = self.max_sim_sum 
        self.metrics_dict["max_sim_mean"] = self.max_sim_sum/self.num_lead_molecules_seen
        
        return self.metrics_dict
