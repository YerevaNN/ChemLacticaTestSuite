from typing import List
import random
import math
import sys
from pathlib import Path
import random
from dataclasses import dataclass
import torch
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.QED import qed

# Disable RDKit logs
RDLogger.DisableLog('rdApp.*')


def query_molecule_properties(model, tokenizer, smiles, property_name, max_new_tokens=15):
    property_start_tag, property_end_tag = f"[{property_name}]", f"[/{property_name}]"
    prompts = [f"</s>[START_SMILES]{smiles}[END_SMILES][{property_name}]"]
    data = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    del data["token_type_ids"]
    outputs = model.generate(
        **data,
        do_sample=False,
        max_new_tokens=max_new_tokens
    )
    predicted_property_values = []
    output_texts = tokenizer.batch_decode(outputs)
    for output_text in output_texts:
        start_ind = output_text.find(property_start_tag)
        end_ind = output_text.find(property_end_tag)
        if start_ind != -1 and end_ind != -1:
            predicted_property_values.append(output_text[start_ind+len(property_start_tag):end_ind])
        else:
            predicted_property_values.append(None)
    return predicted_property_values


def generate_random_number(lower, upper):
    return lower + random.random() * (upper - lower)


def get_short_name_for_ckpt_path(chpt_path: str, hash_len: int=6):
    get_short_name_for_ckpt_path = Path(chpt_path)
    return get_short_name_for_ckpt_path.parent.name[:hash_len] + '-' + get_short_name_for_ckpt_path.name.split("-")[-1]


def canonicalize(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
    # return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), kekuleSmiles=True)


def get_inchi(mol):
    return Chem.MolToInchi(mol)


def get_morgan_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def get_maccs_fingerprint(mol):
    return MACCSkeys.GenMACCSKeys(mol)


def tanimoto_dist_func(mol1, mol2, fingerprint: str="morgan"):
    return DataStructs.TanimotoSimilarity(
        get_morgan_fingerprint(mol1) if fingerprint == 'morgan' else get_maccs_fingerprint(mol1),
        get_morgan_fingerprint(mol2) if fingerprint == 'morgan' else get_maccs_fingerprint(mol2),
    )


def tanimoto_dist_func_with_fings(fing1, fing2):
    return DataStructs.TanimotoSimilarity(fing1, fing2)


def set_seed(seed_value):
    random.seed(seed_value)
    # Set seed for NumPy
    np.random.seed(seed_value)
    # Set seed for PyTorch
    torch.manual_seed(seed_value)


class PMOMoleculeEntry:

    def __init__(
        self,
        smiles: str,
        score: float
    ):
        self.smiles = canonicalize(smiles)
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.inchi = get_inchi(self.mol)
        self.score = score

    def __eq__(self, other):
        return self.inchi == other.inchi

    def __lt__(self, other):
        return self.score < other.score
    
    def __str__(self):
        return f"smiles: {self.smiles}, score: {self.score:.4f}"
    
    def __repr__(self):
        return str(self)


class PMOMoleculeBank:

    def __init__(self, size):
        self.size = size
        self.molecule_entries: List[PMOMoleculeEntry] = []

    def add(self, entries: List[PMOMoleculeEntry]):
        assert type(entries) == list
        self.molecule_entries.extend(entries)
        self.molecule_entries.sort(reverse=True)

        # remove doublicates
        new_molecule_list = []
        for mol in self.molecule_entries:
            if len(new_molecule_list) == 0 or new_molecule_list[-1] != mol:
                new_molecule_list.append(mol)

        self.molecule_entries = new_molecule_list[:min(len(new_molecule_list), self.size)]

    def random_subset(self, subset_size):
        rand_inds = np.random.permutation(min(len(self.molecule_entries), subset_size))
        return [self.molecule_entries[i] for i in rand_inds]