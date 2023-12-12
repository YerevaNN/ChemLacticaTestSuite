from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import RDConfig, QED
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import rdkit.Chem as Chem
import numpy as np
import sys
import os

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def get_canonical(mols):
    if not isinstance(mols, list):
        mols = [mols]
    canonicals = []
    for smiles in mols:
        if smiles != '':
            try:
                mol = Chem.MolFromSmiles(smiles)
                canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
                canonicals.append(canonical_smiles)
            except:
                pass        
    return canonicals

def get_sas(mols):
    if mols == []:
        return [None]
    if not isinstance(mols, list):
        mols = [mols]
    scores = []
    for mol in mols:
        if mol != '':
            try:
                mol_source = Chem.MolFromSmiles(mol)
                sas_score = round(sascorer.calculateScore(mol_source), 3)
            except:
                sas_score = None
        else:
           sas_score = None
        scores.append(sas_score)
    return scores

def get_qed(mols):
    if mols == []:
        return [None]
    if mols == []:
        mols = [mols]
    scores = []
    for mol in mols:
        if mol != '':
            try:
                mol_source = Chem.MolFromSmiles(mol)
                qed_score = round(QED.qed(mol_source), 3)
            except:
                qed_score = None
        else:
            qed_score = None
        scores.append(qed_score)
    return scores

def get_clogp(mols):
    if mols == []:
        return [None]
    if not isinstance(mols, list):
        mols = [mols]
    scores = []
    for mol in mols:
        if mol != '':
            try:
                mol_source = Chem.MolFromSmiles(mol)
                logp_score = round(Descriptors.MolLogP(mol_source), 3)
            except:
                logp_score = None
        else:
            logp_score = None
        scores.append(logp_score)
    return scores

def get_weight(mols):
    if mols == []:
        return [None]
    if not isinstance(mols, list):
        mols = [mols]
    scores = []
    for mol in mols:
        if mol != '':
            try:
                mol_source = Chem.MolFromSmiles(mol)
                weights = round(Descriptors.ExactMolWt(mol_source), 3)
            except:
                weights = None
        else:
            weights = None
        scores.append(weights)
    return scores


def get_similarity(out_sm, inp_smiles):
    # scores = []

    # mol1 = Chem.MolFromSmiles(out_sm[0])
    # mol2 = Chem.MolFromSmiles(inp_smiles)
    # maccs1 = list(MACCSkeys.GenMACCSKeys(mol1).ToBitString())
    # maccs2 = list(MACCSkeys.GenMACCSKeys(mol2).ToBitString())

    # maccs1 = np.array(maccs1)
    # maccs2 = np.array(maccs2)

    # intersection = np.sum(np.bitwise_and(maccs1, maccs2), axis=1)
    # similarity = intersection / (
    #     np.sum(maccs1, axis=1) + np.sum(maccs2, axis=1) - intersection
    # )
    # scores.append(np.round(similarity, decimals=3))
    # return scores
    if not isinstance(out_sm, list):
        out_sm = [out_sm]
    scores = []

    for sm, inp_sm in zip(out_sm, inp_smiles):
        try:
            mol_out = Chem.MolFromSmiles(sm)
            mol_in = Chem.MolFromSmiles(inp_sm)
            inp_fp = Chem.RDKFingerprint(mol_in)
            out_fp = Chem.RDKFingerprint(mol_out)
            similarity = DataStructs.TanimotoSimilarity(inp_fp, out_fp)
            tanimoto = np.round(similarity, decimals=3)
        except:
            tanimoto = None
        scores.append(tanimoto)
    return scores