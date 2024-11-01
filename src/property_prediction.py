import torch
from transformers import OPTForCausalLM, AutoModelForCausalLM, AutoTokenizer
# from chemlactica.utils.utils import get_tokenizer
from rdkit.Chem import Descriptors

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import RDConfig, MACCSkeys, QED
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
from sklearn import metrics
from scipy.stats import spearmanr
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit import Chem, DataStructs
import sys
import os
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

global TIC_DATA 
TIC_DATA = {
        "QED": .01,
        "SAS": .1,
        "TPSA": 1,
        "WEIGHT": 12,
        "CLOGP": .25,
        "Similarity": 0,
        "Similarity CG": 0
        }

def load_validation_mols():
    mols_file = open("/auto/home/menuab/code/ChemLacticaTestSuite/src/stats_data/property_prediction_mols_100.pkl", 'rb')
    mols = pickle.load(mols_file)
    mols_file.close()
    return mols

def load_pubchem_stats():
    pubchem_stats_file = open("/auto/home/menuab/code/ChemLacticaTestSuite/src/stats_data/pubchem_stats.pkl", 'rb')
    pubchem_stats = pickle.load(pubchem_stats_file)
    pubchem_stats_file.close()
    return pubchem_stats

def load_model_tokenizer(model_path, tokenizer_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens = False
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print(f"model loaded on {model.device}, with dtype {model.dtype}, tokenizer with length {len(tokenizer)} loaded")
    return model, tokenizer

def calculate_tanim_sim(m, rel_m):
    m=Chem.MolFromSmiles(m)
    rel_m =Chem.MolFromSmiles(rel_m)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    rel_fp = AllChem.GetMorganFingerprintAsBitVect(rel_m, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp, rel_fp)
    # generate_plot(ground_truths, gens, diffs, property, rmse, mape, correlation, invalids, len(ground_truths), (min(ground_truths), max(ground_truths)), TIC_DATA[property], pubchem_stats, results_path, pdf_export)
    # generate_plot(ground_truths, gens, diffs, property, rmse, mape, correlation, invalids, len(ground_truths), (min(ground_truths), max(ground_truths)), tic[property])
def generate_plot(target_clean, generated_clean, diffs, test_name, rmse,
                  mape, correlation, n_invalid_generations, n_total_gens,
                  pubchem_stats, results_path, pdf_export):
    thickness = TIC_DATA[test_name]
    max_, min_, max_g = np.max(target_clean), np.min(target_clean), np.max(generated_clean)
    # title = f'model_1b_19k_6d68 {test_name} Greedy sampling\n'\
    #         f'{n_invalid_generations}/{n_total_gens} invalid SMILES\n'\
    #         f'rmse {rmse:.3f} mape {mape:.3f} corr: {correlation:.3f}\n'\
    if test_name == "Similarity CG":
        title = f'Similarity Conditional Generation (greedy sampling)\n'\
                f'{n_invalid_generations}/{n_total_gens} invalid SMILES\n'\
                f'rmse {rmse:.3f} mape {mape:.3f} corr: {correlation:.3f}\n'
    else:
        title = f'{test_name} Property Prediction (greedy sampling)\n'\
                f'{n_invalid_generations}/{n_total_gens} invalid SMILES\n'\
                f'rmse {rmse:.3f} mape {mape:.3f} corr: {correlation:.3f}\n'

    fig, ax1 = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(5)
    fig.set_linewidth(2)
    if thickness != 0:    
        ax2 = ax1.twinx()
        stats = pubchem_stats[test_name.upper()]
        ax2.bar([interval.mid for interval in stats.index], stats, width=thickness, alpha=0.3) 
    dist = max_ - min_
    margin = 0.05
    ax1.set_xlim([min_- margin*dist, max_ + margin*dist])
    ax1.scatter(target_clean, generated_clean, c='b')
    # ax1.vlines(nones, ymin=min_, ymax=max_, color='r', alpha=0.3)
    ax1.plot([min_, max_], [min_, max_], color='grey', linestyle='--', linewidth=2)
    if test_name == "Similarity CG":
        ax1.plot(np.arange(0.2,1.05,0.05), np.convolve(diffs, np.ones(3)/3, mode='same'), color='m', alpha=0.5)
        ax1.set_xlabel(f'Target Similarity')
        ax1.set_ylabel(f'Generated Similarity')
    else:
        ax1.plot(target_clean, np.convolve(diffs, np.ones(5)/5, mode='same'), color='m', alpha=0.5)
        ax1.set_xlabel(f'Ground truth {test_name}')
        ax1.set_ylabel(f'Predicted {test_name}')
    ax1.grid(True)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(f'{results_path}/{test_name}_property.png', dpi=300, format="png")
    if pdf_export:    
        fig.savefig(f'{results_path}/{test_name}_property.pdf', dpi=300, format="pdf")
    fig.clf()
    plt.close()

def property_prediction(properties, model, tokenizer, gen_config, pubchem_stats, results_path, log_file, pdf_export, mols):
    results = {}
    for property in properties:
        ground_truths, gens, diffs = [],[],[]
        invalids = 0
        end_property_token = tokenizer.encode(f"[/{property}]", add_special_tokens=False)[0]
        for s in mols:
            prompt = f"{gen_config['separator_token']}{gen_config['mol_token']}{s}{gen_config['end_mol_token']}[{property}]"
            prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            out = model.generate(prompt.input_ids, 
                                 do_sample=False, 
                                 eos_token_id=end_property_token,
                                 max_new_tokens=300)
            out = tokenizer.batch_decode(out)[0]
            try:
                if out.find(f"[{property}]")!=-1:
                    gen_score = float(out[out.find(f"[{property}]") + len(f"[{property}]"):out.find(f"[/{property}]")])
                    if property == "TPSA":
                        gt_score = AllChem.CalcTPSA(Chem.MolFromSmiles(s))
                    elif property == "QED":
                        gt_score = QED.qed(Chem.MolFromSmiles(s))
                    elif property == "SAS":
                        gt_score = sascorer.calculateScore(Chem.MolFromSmiles(s))
                    elif property == "WEIGHT":
                        gt_score = Descriptors.ExactMolWt(Chem.MolFromSmiles(s))
                    elif property == "CLOGP":
                        gt_score = Descriptors.MolLogP(Chem.MolFromSmiles(s))
                    diff = abs(gt_score - gen_score)
                    log_file.write(f"GT: {round(gt_score,2)}, Gen: {gen_score}, diff: {round(diff,2)} {s} {out}\n")
                    ground_truths.append(gt_score)
                    gens.append(gen_score)
                    diffs.append(diff)
                else:
                    log_file.write(f"GT: {gt_score} GEN: {gen_score} {out}\n")
            except:
                log_file.write(f"GT: {gt_score} {out}\n")
                invalids += 1
                raise
            log_file.write("-------------------------------\n")

        combined = list(zip(ground_truths, gens, diffs))
        combined.sort(key=lambda x: x[0])
        ground_truths, gens, diffs = zip(*combined)
        rmse = metrics.mean_squared_error(ground_truths, gens, squared=False)
        mape = metrics.mean_absolute_percentage_error(ground_truths, gens)
        correlation, pvalue = spearmanr(ground_truths, gens)
        generate_plot(ground_truths, gens, diffs, property, rmse, mape, correlation, 
                      invalids, len(ground_truths), pubchem_stats, results_path, pdf_export)
        res = {"rmse": rmse, "mape": mape, "correlation": correlation, "invalids": invalids}
        results[property] = res
        print(f"finished property prediction for {property}")
    return results

def similarity_prediction(model, tokenizer, gen_config, pubchem_stats, results_path, log_file, pdf_export, mols):
    ground_truths, gens, diffs = [],[],[]
    invalids = 0
    end_smiles_token = tokenizer.encode("[/SIMILAR]", add_special_tokens=False)[0]
    for s in mols[:10]:
        for s2 in mols[10:20]:
            prompt = f"{gen_config['separator_token']}{gen_config['mol_token']}{s}{gen_config['end_mol_token']}[SIMILAR]{s2} "
            len_prompt = len(s2) + 1
            prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            out = model.generate(prompt.input_ids, do_sample=False, eos_token_id=end_smiles_token, max_new_tokens=300)
            out = tokenizer.batch_decode(out)[0]
            try:
                if out.find("[/SIMILAR]")!=-1:
                    gen_score = round(float(out[out.find("[SIMILAR]") + len("[SIMILAR]") + len_prompt:out.find("[/SIMILAR]")]), 2)
                    gt_score = calculate_tanim_sim(s, s2)
                    diff = abs(gt_score - gen_score)
                    log_file.write(f"GT: {round(gt_score,2)}, Gen: {gen_score}, diff: {round(diff,2)} {s} {out}\n")
                    ground_truths.append(gt_score)
                    gens.append(gen_score)
                    diffs.append(diff)
                else:
                    log_file.write(f"GT: {gt_score} GEN: {gen_score} {out}\n")
            except:
                log_file.write(f"\n*********s: {s}, s2 {s2}, {out}\n")
                invalids += 1
                raise
            log_file.write("-------------------------------\n")

    combined = list(zip(ground_truths, gens, diffs))
    combined.sort(key=lambda x: x[0])
    ground_truths, gens, diffs = zip(*combined)
    rmse = metrics.mean_squared_error(ground_truths, gens, squared=False)
    mape = metrics.mean_absolute_percentage_error(ground_truths, gens)
    correlation, pvalue = spearmanr(ground_truths, gens)
    generate_plot(ground_truths, gens,diffs, 'Similarity', rmse, mape, correlation, invalids, 
                  len(ground_truths), pubchem_stats, results_path, pdf_export)

    res = {"rmse": rmse, "mape": mape, "correlation": correlation, "invalids": invalids}
    
    return  res

def similarity_generation(model, tokenizer, gen_config, pubchem_stats, results_path, log_file, pdf_export, mols, N_mols):
    ground_truths, gens, diffs = [],[],[]
    invalids = 0
    end_smiles_token = tokenizer.encode(gen_config['end_mol_token'], add_special_tokens=False)[0]
    for s in mols[:N_mols]:
        for sim in np.arange(0.2,1.05,0.05):
            prompt = f"{gen_config['separator_token']}[SIMILAR]{s} {sim:.2f}[/SIMILAR]"
            prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            out = model.generate(prompt.input_ids,
                                 **gen_config["config"])
            out = tokenizer.batch_decode(out.sequences)[0]
            if out.find(gen_config["end_mol_token"])!=-1:
                try:
                    mol = out[out.find(gen_config["mol_token"]) + len(gen_config["mol_token"]):out.find(gen_config["end_mol_token"])]
                    gen_score = calculate_tanim_sim(s, mol)
                    diff = abs(sim - gen_score)
                    log_file.write(f"GT: {round(gen_score,2)}, Gen: {gen_score}, diff: {round(diff,2)} {s} {out}\n")
                    ground_truths.append(sim)
                    gens.append(gen_score)
                    diffs.append(diff)
                except:
                    log_file.write(f"GT: {sim} {out}\n")
                    invalids += 1
                    ground_truths.append(sim)
                    gens.append(0)
                    diffs.append(0)
            else:
                log_file.write(f"GT: {sim} {out}\n")
                invalids += 1
                ground_truths.append(sim)
                gens.append(gen_score)
                diffs.append(0)
            log_file.write("-------------------------------\n")

    combined = list(zip(ground_truths, gens, diffs))
    combined.sort(key=lambda x: x[0])
    ground_truths, gens, diffs = zip(*combined)
    d = np.mean(np.array(diffs).reshape(-1,N_mols),axis=1)
    rmse = metrics.mean_squared_error(ground_truths, gens, squared=False)
    mape = metrics.mean_absolute_percentage_error(ground_truths, gens)
    correlation, pvalue = spearmanr(ground_truths, gens)
    generate_plot(ground_truths, gens, d,'Similarity CG', rmse, mape, correlation, invalids, 
                  len(ground_truths), pubchem_stats, results_path, pdf_export)

    res = {"rmse": rmse, "mape": mape, "correlation": correlation, "invalids": invalids}
    
    return  res

def run_property_predictions(model_path, tokenizer_path, device, properties, generation_config, results_path, log_file, pdf_export):
    model, tokenizer = load_model_tokenizer(model_path, tokenizer_path, device)
    pubchem_stats = load_pubchem_stats()
    validation_mols = load_validation_mols()
    if properties == "all":
        properties = ["TPSA","WEIGHT","QED","SAS","CLOGP"]
        pp_results = property_prediction(properties, model, tokenizer, generation_config, 
                                         pubchem_stats, results_path, log_file, pdf_export, validation_mols)
        sim_pp_results = similarity_prediction(model, tokenizer, generation_config, 
                                               pubchem_stats, results_path, log_file, pdf_export, validation_mols)
        sim_cg_results = similarity_generation(model, tokenizer, generation_config, 
                                               pubchem_stats, results_path, log_file, pdf_export, validation_mols, 20)
    elif properties == 'similarity':
        sim_pp_results = similarity_prediction(model, tokenizer, generation_config, 
                                               pubchem_stats, results_path, log_file, pdf_export, validation_mols)
        sim_cg_results = similarity_generation(model, tokenizer, generation_config, 
                                               pubchem_stats, results_path, log_file, pdf_export, validation_mols, 20)
    else:
        pp_results = property_prediction([properties.upper()], model, tokenizer, generation_config, 
                                         pubchem_stats, results_path, log_file, pdf_export, validation_mols)

    return pp_results, sim_pp_results, sim_cg_results