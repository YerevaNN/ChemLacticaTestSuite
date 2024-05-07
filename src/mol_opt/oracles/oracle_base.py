from abc import ABC, abstractmethod
from chemlactica.mol_opt.utils import MoleculeEntry

class BaseOptimizationOracle(ABC):
    def __init__(self,max_oracle_calls: int,takes_entry=False):
        self.max_oracle_calls = max_oracle_calls
        self.takes_entry = takes_entry
        self.mol_buffer = {}
        self.num_oracle_calls = 0
        self.metrics_dict = {}

    def __call__(self, mol_entry):
        # This function represents logic which is agnostic to the oracle scoring, steps are as follows
        # 1. Check if the molecule has been seen before, in that case just return the prior calculated score
        # 2. Calculate the score, NOTE: this requires a custom implementation for each oracle
        # Note, if any error is encountered, we just return a 0 score
        # 3. Update the molecule buffer with the calculated score
        # 4. Log intermediate results if logging conditon met (logging functionality is user defined)
        # 5. return the value


        if self.mol_buffer.get(mol_entry.smiles):
            return self.mol_buffer[mol_entry.smiles][0]
        try:
            oracle_score = self._calculate_score(mol_entry)
        except Exception as e:
            print(e)
            oracle_score = 0

        self.mol_buffer[mol_entry.smiles] = [oracle_score, len(self.mol_buffer) + 1]
        self.num_oracle_calls += 1

        if self.should_log:
            self._log_intermediate()

        return oracle_score
    
    def reset(self):
        # If you want to ensure an identical configuration for multiple optimizations this can be called between optimizations.
        # This can be useful for e.g. metric tracking, the needed data can be implemented via _store_custom_data.

        self.mol_buffer = {}
        self.num_oracle_calls = 0
        self._store_custom_data()
        self._clear_custom_data()

    @abstractmethod
    def _store_custom_data(self):
        pass

    @abstractmethod
    def _clear_custom_data(self):
        pass

    @abstractmethod
    def _calculate_score(self, mol_entry):
        pass

    @property
    def should_log(self):
        return self.num_oracle_calls % 500 == 0

    @abstractmethod
    def _log_intermediate(self):
        pass

    @abstractmethod
    def _calculate_metrics(self):
        pass

    @property
    def budget(self):
        return self.max_oracle_calls

    @property
    def finish(self):
        return self.num_oracle_calls >= self.max_oracle_calls

