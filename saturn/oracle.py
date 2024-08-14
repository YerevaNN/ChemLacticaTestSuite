import numpy as np
from typing import List, Union
from rdkit.Chem import QED

from chemlactica.mol_opt.utils import MoleculeEntry

from saturn.oracles.docking.geam_oracle import GEAMOracle
from saturn.oracles.synthesizability.sascorer import calculateScore
from saturn.oracles.dataclass import OracleComponentParameters


class SaturnDockingOracle:
    def __init__(self, max_oracle_calls: int, target: str, takes_entry: bool = False):
        # maximum number of oracle calls to make
        self.max_oracle_calls = max_oracle_calls

        # the frequence with which to log
        self.freq_log = 100

        # the buffer to keep track of all unique molecules generated
        self.mol_buffer = {}
        self.mol_component_scores_buffer = {}

        # the maximum possible oracle score or an upper bound
        self.max_possible_oracle_score = 1.0

        # if True the __call__ function takes list of MoleculeEntry objects
        # if False (or unspecified) the __call__ function takes list of SMILES strings
        self.takes_entry = takes_entry
        self.reports_component_scores = True

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

        # In case there is a . (dot) sign in SMILES string, it implies
        # that 2 or more molecules are present in the SMILES string, which does not work for this docking oracle
        # multi_molecules = [mol for mol in molecules if '.' in mol.smiles]
        # dummy_scores = [[0, {}]] * len(multi_molecules)
        # molecules = [mol for mol in molecules if '.' not in mol.smiles]

        new_molecules = []
        full_scores = []

        for mol in molecules:
            if self.mol_buffer.get(mol.smiles):
                full_scores.append([self.mol_buffer[mol.smiles][0], self.mol_component_scores_buffer[mol.smiles]])
                continue

            new_molecules.append(mol)

        if len(new_molecules) > 0:
            rdkit_mols = [mol.mol for mol in new_molecules]
            raw_vina_scores, *_, aggregated_scores = self._saturn_oracle(rdkit_mols)
            # Reverse the scores, as our optimization algorithm is a maximization algorithm
            raw_vina_scores = [-score for score in raw_vina_scores]
            raw_sa_scores = [calculateScore(mol) for mol in rdkit_mols]
            raw_qed_scores = [QED.qed(mol) for mol in rdkit_mols]

            new_scores = [[
                agg,
                {"vina": vina, "qed": qed, "sa": sa}
            ] for agg, vina, qed, sa in zip(aggregated_scores, raw_vina_scores, raw_qed_scores, raw_sa_scores)]
            full_scores.extend(new_scores)

            for score, mol in zip(new_scores, new_molecules):
                agg, components = score
                self.mol_buffer[mol.smiles] = [agg, len(self.mol_buffer) + 1]
                self.mol_component_scores_buffer[mol.smiles] = components

                if len(self.mol_buffer) % self.freq_log == 0:
                    self.log_intermediate()

        # return dummy_scores + full_scores
        return full_scores
    
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