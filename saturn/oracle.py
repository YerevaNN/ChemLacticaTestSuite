import numpy as np
from typing import List, Union
from rdkit.Chem import MolFromSmiles

from chemlactica.mol_opt.utils import MoleculeEntry

from saturn.oracles.docking.geam_oracle import GEAMOracle
from saturn.oracles.dataclass import OracleComponentParameters


class SaturnDockingOracle:
    def __init__(self, max_oracle_calls: int, target: str, takes_entry: bool = False):
        # maximum number of oracle calls to make
        self.max_oracle_calls = max_oracle_calls

        # the frequence with which to log
        self.freq_log = 100

        # the buffer to keep track of all unique molecules generated
        self.mol_buffer = {}

        # the maximum possible oracle score or an upper bound
        self.max_possible_oracle_score = 1.0

        # if True the __call__ function takes list of MoleculeEntry objects
        # if False (or unspecified) the __call__ function takes list of SMILES strings
        self.takes_entry = takes_entry

        params = OracleComponentParameters(
            name='geam',
            specific_parameters={
                'target': target,
            },
            reward_shaping_function_parameters={
                "transformation_function": "sigmoid",
                "parameters": {
                    "low": 75,
                    "high": 350,
                    "k": 0.15
                }
                }
        )
        self._saturn_oracle = GEAMOracle(params)

    def __call__(self, molecules: List[Union[str, MoleculeEntry]]) -> List[float]:
        """
            Evaluate and return the oracle scores for molecules. Log the intermediate results if necessary.
        """
        if not self.takes_entry:
            molecules = [MoleculeEntry(smiles) for smiles in molecules]


        new_molecules = []
        oracle_scores = []
        for mol in molecules:
            # if len(self.mol_buffer) % self.freq_log == 0:
            #     self.log_intermediate()

            if self.mol_buffer.get(mol.smiles):
                oracle_scores.append(sum(self.mol_buffer[mol.smiles][0]))
                continue

            new_molecules.append(mol)

        rdkit_mods = [mol.mol for mol in new_molecules]
        new_oracle_scores, *_ = self._saturn_oracle(rdkit_mods)
        oracle_scores.extend(new_oracle_scores)

        for score, mol in zip(new_oracle_scores, new_molecules):
            self.mol_buffer[mol.smiles] = [score, len(self.mol_buffer) + 1]
            if len(self.mol_buffer) % self.freq_log == 0:
                self.log_intermediate()

        # get the oracle scores
        # oracle_scores = self._saturn_oracle(rdkit_modls) 
        return oracle_scores
    
    def log_intermediate(self):
        scores = [v[0] for v in self.mol_buffer.values()][-self.max_oracle_calls:]
        scores_sorted = sorted(scores, reverse=True)[:100]
        n_calls = len(self.mol_buffer)

        score_avg_top1 = np.max(scores_sorted)
        score_avg_top10 = np.mean(scores_sorted[:10])
        score_avg_top100 = np.mean(scores_sorted)

        print(f"{n_calls}/{self.max_oracle_calls} | ",
                f'avg_top1: {score_avg_top1:.3f} | '
                f'avg_top10: {score_avg_top10:.3f} | '
                f'avg_top100: {score_avg_top100:.3f}')

    def __len__(self):
        return len(self.mol_buffer)

    @property
    def budget(self):
        return self.max_oracle_calls

    @property
    def finish(self):
        # the stopping condition for the optimization process
        return len(self.mol_buffer) >= self.max_oracle_calls