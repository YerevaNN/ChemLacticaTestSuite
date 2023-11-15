import re
import os
import glob
import time
import json
import pickle
from itertools import chain
from datetime import datetime

import numpy as np
from scipy.stats import spearmanr
from sklearn import metrics
import matplotlib.pyplot as plt
from aim import Run, Image
import torch
from torch import bfloat16, float32
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import mol_util
from custom_modeling_opt import CustomOPTForCausalLM
from property_2_Mol_config import evaluation_configs
from pubchem_checker.check_in_pubchem import check_in_pubchem
# from assert_tokenizer import assert_tokenizer

seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Property2Mol:
    def __init__(
            self,
            test_suite,
            property_range,
            generation_config,
            regexp,
            model_checkpoint_path,
            tokenizer_path,
            torch_dtype,
            device,
            top_N,
            n_per_vs_rmse,
            include_eos,
            check_for_novelty,
            track,
            description,
            ) -> None:
        
        self.test_suite=test_suite
        self.property_range=property_range
        self.generation_config=generation_config["config"]
        self.generation_config_name=generation_config["name"]
        self.regexp=regexp
        self.model_checkpoint_path = model_checkpoint_path + '/' if model_checkpoint_path[-1] != '/' else model_checkpoint_path
        self.tokenizer_path=tokenizer_path
        self.torch_dtype=torch_dtype
        self.device = device
        self.track = track
        
        self.smiles_prefix = "[START_SMILES]"
        self.eos_string = "</s>"
        self.include_eos = include_eos
        self.top_N = generation_config.get("top_N", 0)
        # self.top_N = top_N
        self.n_per_vs_rmse = n_per_vs_rmse

        self.molecules_set = set()
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        if self.track:
            self.start_aim_tracking(description)
        else:
            self.eval_hash = 'none'
        self.start_log_file()
        # assert_model_tokenizer()
        self.pubchem_stats = self.get_pubchem_stats()
        self.check_for_novelty = check_for_novelty
        
    @staticmethod
    def get_pubchem_stats():
        pubchem_stats_file = open("src/pubchem_stats.pkl", 'rb')
        pubchem_stats = pickle.load(pubchem_stats_file)
        pubchem_stats_file.close()
        return pubchem_stats

    def start_aim_tracking(self, description):
        self.aim_run = Run(experiment=description) if self.track else None
        self.eval_hash = self.aim_run.hash if self.aim_run else 'none'
        try:
            training_args = vars(torch.load(self.model_checkpoint_path + '/training_args.bin'))
            evaluation_config['learning_rate'] = training_args['learning_rate']
            evaluation_config['output_dir'] = training_args['output_dir']
            evaluation_config['per_device_train_batch_size'] = training_args['per_device_train_batch_size']
            evaluation_config['weight_decay'] = training_args['weight_decay']
            evaluation_config['max_steps'] = training_args['max_steps']
            evaluation_config['model_hash'] = training_args['output_dir'].split('/')[-1]
        except:
            pass
        self.aim_run['hparams'] = evaluation_config
    
    def start_log_file(self):
        model_name = self.model_checkpoint_path.split("/")[-2]
        self.results_path = os.path.join(f"/home/menuab/code/ChemLacticaTestSuite/results/property_2_Mol/"\
                                         f"{datetime.now().strftime('%Y-%m-%d-%H:%M')}-{model_name}"\
                                         f"-{self.generation_config_name}-{self.eval_hash}/")
        print(f'results_path = {self.results_path}\n')
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        self.log_file = open(self.results_path + 'full_log.txt', 'w+')
        
        self.log_file.write(f'results of property to molecule test performed at '\
                            f'{datetime.now().strftime("&Y-%m-%d, %H:%M")}\n')
        self.log_file.write(f'evaluation config: \n{json.dumps(evaluation_config, indent=4)}\n')

    def load_model(self):
        # model = AutoModelForCausalLM.from_pretrained(self.model_checkpoint_path, torch_dtype=self.torch_dtype)
        model = CustomOPTForCausalLM.from_pretrained(
            self.model_checkpoint_path,
            use_flash_attn=True,
            torch_dtype=getattr(torch, self.torch_dtype)
            )
        model.eval()
        model.to(self.device)
        print(f'model loaded with embedding size of : {model.model.decoder.embed_tokens.num_embeddings}')

        return model

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        print(f"tokenizer loaded with size of {len(tokenizer)}")

        return tokenizer

    def get_inputs(self, properties):
        inputs_ = []
        for property in properties:
            property_range = self.property_range[property]["range"]
            property_step = self.property_range[property]["step"]
            inputs_.append(np.round(np.arange(property_range[0],
                                              property_range[1] + property_step,
                                              property_step), 3))
        
        inputs = []
        for value in inputs_[0]:
            if len(self.tokenizer) == 50028:
                input = f'[{property.upper()} {value}]{self.smiles_prefix}'
            else:
                input = f'[{property.upper()}]{value}[/{property.upper()}]{self.smiles_prefix}'
            if self.include_eos:
                input = self.eos_string + input
            inputs.append(input)

        return inputs

    def get_targets(self, properties):
        targets = []
        for property in properties:
            property_range = self.property_range[property]["range"]
            property_step = self.property_range[property]["step"]
            targets = [[t] for t in np.round(np.arange(property_range[0],
                                                        property_range[1] + property_step,
                                                        property_step), 3)]
        # targets_ = np.meshgrid(targets_) TODO:develop later for multiple targets

        return targets

    @staticmethod
    def get_property_fns(properties):
        property_fns = [getattr(mol_util, 'get_' + property.lower()) for property in properties]

        return property_fns

    def generate_outputs(self):
        input_ids = [self.tokenizer(input, return_tensors="pt").to(self.device).input_ids for input in self.inputs]
        outputs, raw_outputs, perplexities_list, token_lengths, norm_logs_list, self.molecules_set = [], [], [], [], [], set()
        for input_id in input_ids:
            context_length = input_id.shape[1]
            perplexities, texts, lengths, norm_log, out = [], [], [], [], []
            range_ = 10 if self.generation_config["num_return_sequences"] == 100 else 1
            for _ in range(range_):
                output = self.model.generate(
                    input_ids=input_id,
                    **self.generation_config   
                )
                if self.generation_config["num_beams"] > 1:
                    scores = self.model.compute_transition_scores(
                            sequences=output.sequences,
                            scores=output.scores,
                            beam_indices=output.beam_indices,
                            normalize_logits=True
                        ).cpu().detach()
                else:
                    scores = self.model.compute_transition_scores(
                            sequences=output.sequences,
                            scores=output.scores,
                            normalize_logits=True
                        ).cpu().detach()
                end_smiles = np.nonzero(output.sequences==20).cpu().numpy() # 20 for END_SMILES token
                end_smiles_indices = end_smiles[np.unique(end_smiles[:, 0], return_index=True)[1]]
                perplexities_ = [round(np.exp(-scores[index[0], :index[1] - context_length + 1].mean().item()), 4)
                                for index in end_smiles_indices]
                norm_logs = [round(scores[index[0], :index[1] - context_length + 1].sum().item(), 4) 
                            for index in end_smiles_indices]
                # print(norm_logs)
                # perplexities = np.exp(-np.ma.masked_invalid(scores.cpu().numpy()).mean(axis=1).data)
                if self.generation_config["do_sample"] == True:
                    sorted_outputs = sorted(zip(perplexities_,
                                                norm_logs,
                                                output.sequences[end_smiles_indices[:, 0]],
                                                end_smiles_indices[:, 1] - context_length + 1),
                                                key=lambda x: x[0])[:self.top_N]
                    
                    for perplexity, n_log, output, len_ in sorted_outputs:
                        norm_log.append(n_log)
                        texts.append(self.tokenizer.decode(output[context_length:]))
                        perplexities.append(perplexity)
                        lengths.append(len_)
                else:
                    lengths = [output.sequences.shape[-1] - context_length]
                    texts = [self.tokenizer.decode(out[context_length:]) for out in output.sequences]
                    norm_log = [norm_logs]
                out = []
                for text in texts:
                    try:    
                        captured_text = re.match(self.regexp, text).group()
                        if captured_text not in self.molecules_set:
                            self.molecules_set.add(captured_text)
                    except:
                        captured_text = ''
                    out.append(captured_text)
                # print(_, len(out))
            outputs.append(out)
            raw_outputs.append(texts)
            perplexities_list.append(perplexities)
            token_lengths.append(lengths)
            norm_logs_list.append(norm_log)
            # print('gen time',len(out), len(outputs[-1]), len(raw_outputs[-1]), len(perplexities_list[-1]), len(token_lengths[-1]), len(norm_logs_list[-1]))
            
        return outputs, raw_outputs, perplexities_list, token_lengths, norm_logs_list
    
    def calculate_properties(self, property_fns):
        # TODO: drop the hard coded index and adjust for multiple targets
        calculated_properties = [property_fns[0](out) for out in self.outputs]

        return calculated_properties

    def get_stats(self):
        errors = []
        n_invalid_generations = 0
        # print('get stats lengs', len(self.calculated_properties), len(self.targets))
        for c_property, target in zip(self.calculated_properties, self.targets):
            error = []
            for prop in c_property:
                if prop != None:
                    error.append(round(abs(prop - target[0]), 2))
                else:
                    error.append(0)
                    n_invalid_generations += 1
            errors.append(error)
        errors.append(error)
        uniques = set(mol_util.get_canonical(list(chain(*self.outputs))))
        n_uniques = len(uniques)
        if self.check_for_novelty:
            in_pubchem = check_in_pubchem(uniques)
            n_in_pubchem = sum(in_pubchem.values())
        else:
            n_in_pubchem = 0
        n_total_gens = len(self.inputs) * self.generation_config['num_return_sequences']
        return errors, n_invalid_generations, n_uniques, n_in_pubchem, n_total_gens

    def write_to_file(self, test_name):
        self.log_file.write(f'properties under test: {test_name}\n')
        self.log_file.write(f'number of total generations: {self.n_total_gens}\n')
        self.log_file.write(f'number of unique molecules generated: {self.n_unique}\n')
        self.log_file.write(f'number of in pubchem molecules generated: {self.n_in_pubchem}\n')
        self.log_file.write(f"No valid SMILES generated in {self.n_invalid_generations} out of"\
                            f" {len(self.targets)} cases\n----------\n\n")

        for items in zip(self.inputs, self.targets, self.outputs, self.raw_outputs,
                        self.calculated_properties, self.errors, self.perplexities,
                        self.token_lengths, self.norm_logs):
            input, target, output, raw_output, c_prop, err, perplexity, length, n_logs = items
            self.log_file.write(f'input: {input}\n')
            self.log_file.write(f'target value: {target[0]}\n')
            for r in raw_output:
                self.log_file.write(f'raw_output: {r}\n')
            self.log_file.write('-----------\n')
            for o, cp, e, per, l, nl in zip(output, c_prop, err, perplexity, length, n_logs):
                self.log_file.write(f'captured_output: {o}\n')
                self.log_file.write(f'generated_property: {cp} diff: {e}, '\
                                    f'perplexity: {per}, normalized logs sum: {nl} '\
                                    f'token length: {l}, char length: {len(o)}\n')
            self.log_file.write('***********\n')

    def clean_outputs(self):
        target_clean, generated_clean, nones = [], [], []
        for target, c_props in zip(self.targets, self.calculated_properties):
            target *= self.generation_config["num_return_sequences"]
            for t, cp in zip(target, c_props):
                if  cp != None:
                    target_clean.append(t)
                    generated_clean.append(cp)
                else:
                    nones.append(t)
        correlation, pvalue = spearmanr(target_clean, generated_clean)
        rmse = metrics.mean_squared_error(target_clean, generated_clean, squared=False)
        mape = metrics.mean_absolute_percentage_error(target_clean, generated_clean)
        return target_clean, generated_clean, nones, correlation, rmse, mape

    def generate_plot(self, test_name, target_clean, generated_clean, nones, correlation, rmse, mape):
        max_, min_, max_g = np.max(self.targets), np.min(self.targets), np.max(generated_clean)
        title = f'{self.generation_config_name} generation of {test_name} with {self.model_checkpoint_path.split("/")[-2]}\n'\
                f'rmse {rmse:.3f} mape {mape:.3f}\ncorr: {correlation:.3f}, N invalid: {self.n_invalid_generations}, '\
                f'N total: {self.n_total_gens}\n N Unique: {self.n_unique}, N in PubChem: {self.n_in_pubchem}'
        
        fig, ax1 = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)
        fig.set_linewidth(4)
        ax2 = ax1.twinx()
        stats = self.pubchem_stats[test_name.upper()]
        property_range = self.property_range[test_name]["range"]
        stats_width = (property_range[1] - property_range[0]) / 100
        ax2.bar([interval.mid for interval in stats.index], stats, width=stats_width, alpha=0.3) 
        
        # ax1.text(1.2 * max_, 0.90 * max(max_g, max_), f"Spearman correlation: {correlation:.3f}")
        # ax1.text(1.2 * max_, 0.85 * max(max_g, max_), f"N invalid gens: {self.n_invalid_generations}")
        # ax1.text(1.2 * max_, 0.80 * max(max_g, max_), f"N of total gens: {self.n_total_gens}")
        # ax1.text(1.2 * max_, 0.75 * max(max_g, max_), f"N of Unique Mols: {self.n_unique}")
        # ax1.text(1.2 * max_, 0.70 * max(max_g, max_), f"N of in PubChem Mols: {self.n_in_pubchem}")
        ax1.scatter(target_clean, generated_clean, c='b')
        ax1.vlines(nones, ymin=min_, ymax=max_, color='r', alpha=0.3)
        ax1.plot([min_, max_], [min_, max_], color='grey', linestyle='--', linewidth=2)
        ax1.set_xlabel(f'target {test_name}')
        ax1.set_ylabel(f'generated {test_name}')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(self.results_path + test_name + '.png', dpi=300, format="png")
        fig.clf()
        plt.close()

    def generate_length_calibration_plots(self, x_axis_name, test_name, log_norms, lengths, errors, target):
        # print(x_axis_name)
        # print(lengths[:20])
        path = self.results_path + "per_vs_rmse/calibration/"
        if not os.path.exists(path):
            os.makedirs(path)

        len_ranges = [30, 60, 90, 120, 150, 170]
        tol = 2
        x_axis, y_axis, z_axis = [[],[],[],[],[],[],[]], [[],[],[],[],[],[],[]], [[],[],[],[],[],[],[]]
        for lnorm, leng, err in zip(log_norms, lengths, errors):
            for en, lenr in enumerate(len_ranges):
                if leng in range(lenr-tol, lenr+tol+1):
                    x_axis[en].append(lnorm)
                    y_axis[en].append(err)
                    z_axis[en].append(leng)

        fig1, axs1 = plt.subplots(6, 1, figsize=(12, 40))
        fig1.suptitle(f'Calibration Log prob vs. Error with {test_name}={target}')

        for en, lr in enumerate(len_ranges):
            im1 = axs1[en].scatter(x_axis[en], y_axis[en], c=z_axis[en], s=70)
            correlation, pvalue = spearmanr(x_axis[en], y_axis[en])
            axs1[en].set_title(f'length= {lr} +/-{tol}, Spearman correlation= {correlation:.3f}')
            axs1[en].set_xlabel(x_axis_name)
            axs1[en].set_ylabel('Error')
            axs1[en].grid()
            cbar1 = fig1.colorbar(im1, label='Length')

        fig1.savefig(path + f'{test_name}_{target}_calibration_{x_axis_name}_vs_err.png', dpi=300, format="png")
        fig1.clf()

    def generate_perplexity_vs_rmse(self, test_name):
        indices = np.linspace(0, len(self.perplexities) - 1, self.n_per_vs_rmse + 2, dtype=int)[1:self.n_per_vs_rmse + 1]
        for data in [self.norm_logs, self.perplexities]:
            data_name = "Log_probs" if data[0][0] < 0 else "Perplexity"
            
            fig1, axs1 = plt.subplots(1, self.n_per_vs_rmse, figsize=(self.n_per_vs_rmse * 6, 6))
            fig1.suptitle(f'{data_name} vs. Length vs. Absolute Error, overall RMSE={round(self.rmse, 2)} '\
                        f'N samples={self.generation_config["num_return_sequences"]}') 
            
            fig2, axs2 = plt.subplots(1, self.n_per_vs_rmse, figsize=(self.n_per_vs_rmse * 6, 6))
            fig2.suptitle(f'{data_name} vs. Absolute Error vs. Length, overall RMSE={round(self.rmse, 2)} '\
                        f'N samples={self.generation_config["num_return_sequences"]}')
            
            for en, i in enumerate(indices):
                # print('pre len',i, data_name, len(data[i]), len(self.errors[i]), len(self.token_lengths[i]))
                x_axis, error_clean, invalid, lengths = [], [], [], []
                for x, e, l in zip(data[i], self.errors[i], self.token_lengths[i]):
                    if e > 0:
                        x_axis.append(x)
                        error_clean.append(e)
                        lengths.append(l)
                    else:
                        invalid.append(x)
                # print('post len',i, data_name, len(x_axis), len(error_clean), len(lengths))
                self.generate_length_calibration_plots(data_name, test_name, x_axis, lengths, error_clean, self.targets[i][0])

                im1 = axs1[en].scatter(x_axis, lengths, c=error_clean, s=70)
                axs1[en].set_title(f'{test_name}={self.targets[i][0]}')
                # axs[en].set_xlim((0, 2000))
                # axs[en].set_ylim((0, 300))
                axs1[en].set_xlabel(data_name)
                axs1[en].set_ylabel('Length')
                axs1[en].grid()
                color_max = max(error_clean, default=0)
                cbar1 = fig1.colorbar(im1, label='Error')

                im2 = axs2[en].scatter(x_axis, error_clean, c=lengths, s=70)
                axs2[en].set_title(f'{test_name}={self.targets[i][0]}')
                # axs[en].set_xlim((0, 2000))
                # axs[en].set_ylim((0, 300))
                axs2[en].set_xlabel(data_name)
                axs2[en].set_ylabel('Error')
                axs2[en].grid()
                color_max = max(lengths, default=0)
                cbar2 = fig2.colorbar(im2, label='Length')
            fig1.savefig(self.results_path + "per_vs_rmse/" + f'{test_name}_{data_name}_vs_len.png', dpi=300, format="png")
            fig1.clf()
            fig2.savefig(self.results_path + "per_vs_rmse/" + f'{test_name}_{data_name}_vs_err.png', dpi=300, format="png")
            fig2.clf()

    def track_stats(self, test_name):
        self.aim_run.track(
            {
                "rmse": self.rmse,
                "Spearman correlation": self.correlation,
                "mape": self.mape,
                "N total gens": self.n_total_gens,
                "N invalid gens": self.n_invalid_generations,
                "N of Unique Mols": self.n_unique,
                "N of in PubChem Mols": self.n_in_pubchem,
            },
        context={'subset': test_name}
        )
        image_paths = [path for path in glob.glob(self.results_path + '**', recursive=True) 
                       if '.png' in path and test_name in path] 
        for path in image_paths:
            aim_image = Image(
                    path,
                    format='png',
                    optimize=True,
                    quality=50
                )
            self.aim_run.track(aim_image, name=path.split('/')[-1], context={'subset': test_name})

    def run_property_2_Mol_test(self):    
        for test_name, sample in list(self.test_suite.items()):
            time_start = time.time()
            input_properties = sample["input_properties"]
            target_properties = sample["target_properties"]
            self.targets = self.get_targets(target_properties)
            self.inputs = self.get_inputs(input_properties)
            property_fns = self.get_property_fns(target_properties)
            self.outputs, self.raw_outputs, self.perplexities, self.token_lengths, self.norm_logs = self.generate_outputs()
            # for i in range(len(self.targets)):
            #     print(i, len(self.inputs[i]), len(self.targets[i]), len(self.outputs[i]), len(self.perplexities[i]), len(self.token_lengths[i]), len(self.norm_logs[i]))
            self.calculated_properties = self.calculate_properties(property_fns)
            self.errors, self.n_invalid_generations, self.n_unique, self.n_in_pubchem, self.n_total_gens = self.get_stats()
            target_clean, generated_clean, nones, self.correlation, self.rmse, self.mape = self.clean_outputs()
            self.write_to_file(test_name)
            self.generate_plot(test_name, target_clean, generated_clean, nones, self.correlation, self.rmse, self.mape)
            if self.generation_config["do_sample"] == True:
                path = self.results_path + "per_vs_rmse/"
                if not os.path.exists(path):
                    os.makedirs(path)
                self.generate_perplexity_vs_rmse(test_name)
            if self.track:
                self.track_stats(test_name)
            print(f"finished evaluating test for {test_name}")
            print(f"{len(self.inputs)} samples evaluated in {int(time.time()-time_start)} seconds")


if __name__ == "__main__":
    
    for evaluation_config in evaluation_configs:
        print(f"evaluating model: {evaluation_config['model_checkpoint_path'].split('/')[-2]} "\
              f"with {evaluation_config['generation_config']['name']} config")
        property_2_Mol = Property2Mol(**evaluation_config)
        property_2_Mol.run_property_2_Mol_test()
        property_2_Mol.log_file.close()
        del property_2_Mol