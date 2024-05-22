from abc import ABC, abstractmethod
from chemlactica.mol_opt.utils import MoleculeEntry
from typing import Union, List

class BaseOptimizationOracle(ABC):
    def __init__(self,max_oracle_calls: int,takes_entry=False):
        self.max_oracle_calls = max_oracle_calls
        self.takes_entry = takes_entry
        self.mol_buffer = {}
        self.num_oracle_calls = 0
        self.metrics_dict = {}

    def __call__(self, mol_entries: Union[MoleculeEntry, List[MoleculeEntry]]):
        # This function represents logic which is agnostic to the oracle scoring, steps are as follows
        # 1. Check if the molecule has been seen before, in that case just return the prior calculated score
        # 2. Calculate the score, NOTE: this requires a custom implementation for each oracle
        # Note, if any error is encountered, we just return a 0 score
        # 3. Update the molecule buffer with the calculated score
        # 4. Log intermediate results if logging conditon met (logging functionality is user defined)
        # 5. return the value

        if not isinstance(mol_entries, List):
            mol_entries = [mol_entries]
        oracle_scores = self.retrieve_scores_from_buffer(mol_entries)
        oracle_scores = self.merge_stored_and_calculated_scores(mol_entries, oracle_scores)

        if self.should_log:
            self._log_intermediate()

        return oracle_scores

    def merge_stored_and_calculated_scores(self, mol_entries, scores):
        entry_index_map = {}
        for i, mol_entry in enumerate(mol_entries):
            if scores[i] is None:
                if mol_entry not in entry_index_map:
                    entry_index_map[mol_entry] = i

        entries_to_score = list(entry_index_map.keys())
        new_scores = self._calculate_score(entries_to_score)

        for new_entry, new_score in zip(entries_to_score,new_scores):
            index = entry_index_map[new_entry]
            scores[index] = new_score

            for i in range(index + 1, len(mol_entries)):
                if mol_entries[i] == new_entry and scores[i] is None:
                    scores[i] = new_score

            self.num_oracle_calls += 1
            self.mol_buffer[new_entry.smiles] = (new_score, self.num_oracle_calls)


        return scores

    def retrieve_scores_from_buffer(self, mol_entries):
        scores = []
        for mol_entry in mol_entries:
            buffer_retrieval= self.mol_buffer.get(mol_entry.smiles)
            if buffer_retrieval is not None:
                score = buffer_retrieval[0]
            else:
                score = None
            scores.append(score)


        return scores

    
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

