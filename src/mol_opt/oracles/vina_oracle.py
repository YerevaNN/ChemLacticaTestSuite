from .oracle_base import BaseOptimizationOracle
import requests
import json
import numpy as np
from .oracle_utils import generative_yield, oracle_burden

class VinaOracle(BaseOptimizationOracle):

    def __init__(self, url, max_oracle_calls: int):
        super().__init__(max_oracle_calls,takes_entry=True)
        self.url = url
        self.num_lead_molecules_seen = 0
        self.num_molecules_optimized = 0
        self.max_sim_mean = 0
        self.max_sim_sum = 0
        self.num_optimized = 0

    @property
    def should_log(self):
        return self.num_oracle_calls % 2 == 0

    def set_lead_entry(self, lead_entry):
        self.lead_entry = lead_entry
        self.num_lead_molecules_seen +=1

    def _log_intermediate(self):
        all_values = [v[0] for v in self.mol_buffer.values()]
        scores = sorted(all_values, reverse=True)[:100]
        
        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        
        print(f'{self.num_oracle_calls}/{self.max_oracle_calls} | '
                f'avg_top1: {avg_top1:.3f} | '
                f'avg_top10: {avg_top10:.3f} | '
                f'avg_top100: {avg_top100:.3f}')

        # try:
        print({
            "avg_top1": avg_top1, 
            "avg_top10": avg_top10, 
            "avg_top100": avg_top100,
            "n_oracle": self.num_oracle_calls,
        })

    def _calculate_score(self, mol_entry):
        input_data_json = json.dumps([mol_entry.smiles])
        response = requests.post(self.url, data=input_data_json)
        oracle_score = float(dict(response.json())["result"][0])
        return oracle_score
    
    def _calculate_metrics(self):
        metrics_dict = {}
        metrics_dict['generative_yield_0.7'] = generative_yield(self.mol_buffer,0.7)
        metrics_dict['generative_yield_0.8'] = generative_yield(self.mol_buffer,0.8)

        metrics_dict['oracle_burden_0.8(1)'] = oracle_burden(self.mol_buffer,1,self.num_oracle_calls,0.8)
        metrics_dict['oracle_burden_0.8(10)'] = oracle_burden(self.mol_buffer,10,self.num_oracle_calls,0.8)
        metrics_dict['oracle_burden_0.8(100)'] = oracle_burden(self.mol_buffer,100,self.num_oracle_calls,0.8)
        return metrics_dict

    def _store_custom_data(self):
        pass

    def _clear_custom_data(self):
        pass
