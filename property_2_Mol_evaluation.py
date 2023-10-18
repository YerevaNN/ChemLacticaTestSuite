import numpy as np
import re
import time
import json
import torch
from torch import bfloat16, float32
from transformers import AutoTokenizer, AutoModelForCausalLM
# from assert_tokenizer import assert_tokenizer
from sklearn import metrics
import os
from utils import mol_util
from custom_modeling_opt import CustomOPTForCausalLM
import scipy
from scipy.stats import spearmanr
from datetime import datetime
import matplotlib.pyplot as plt

class Property2Mol:
    def __init__(
            self,
            test_suit,
            property_range,
            generation_config,
            regexp,
            model_checkpoint_path,
            tokenizer_path,
            torch_dtype,
            device,
            include_eos=True,
            ) -> None:
        
        self.test_suit=test_suit
        self.property_range=property_range
        self.generation_config=generation_config
        self.regexp=regexp
        self.model_checkpoint_path = model_checkpoint_path + '/' if model_checkpoint_path[-1] != '/' else model_checkpoint_path
        self.tokenizer_path=tokenizer_path
        self.torch_dtype=torch_dtype
        self.device = device
        
        self.smiles_prefix = "[START_SMILES]"
        self.eos_string = "</s>"
        self.include_eos = include_eos

        self.molecules_set = set()
        self.invalid_generations = {"not_captured":0,  "not_valid":0}

        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.log_file = self.start_log_file()
        # assert_model_tokenizer()

    def start_log_file(self):
        model_name = self.model_checkpoint_path.split("/")[-2]
        self.results_path = os.path.join(f"results/property_2_Mol/{datetime.now().strftime('%Y-%m-%d-%H:%M')}"\
                                    f"-{model_name}/")
        print(f'results_path = {self.results_path}\n')
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        log_file = open(self.results_path + 'full_log.txt', 'w+')
        
        log_file.write(f'results of property to molecule test performed at '\
                            f'{datetime.now().strftime("%Y-%m-%d, %H:%M")}\n')
        log_file.write(f'model checkpoint path: {self.model_checkpoint_path}\n')
        log_file.write(f'property combinations being evaluated: \n{json.dumps(self.test_suit, indent=4)}\n')
        log_file.write(f'generation config: \n{json.dumps(self.generation_config, indent=4)}\n\n')

        return log_file

    def load_model(self):
        # model = AutoModelForCausalLM.from_pretrained(self.model_checkpoint_path, torch_dtype=self.torch_dtype)
        model = CustomOPTForCausalLM.from_pretrained(
            self.model_checkpoint_path,
            use_flash_attn=True,
            torch_dtype=self.torch_dtype
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
            input = f'[{property.upper()} {value}]{self.smiles_prefix}'
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
        outputs = []
        raw_outputs = []
        self.molecules_set = set()
        self.invalid_generations = {"not_captured":0,  "not_valid":0}        
        for input_id in input_ids:
            output = self.model.generate(
                input_ids=input_id,
                **self.generation_config,
            )
            context_length = len(input_id[0])
            texts = [self.tokenizer.decode(out[context_length:]) for out in output]
            raw_outputs.append(texts)
            out = []
            for text in texts:
                try:    
                    captured_text = re.match(self.regexp, text).group()
                    if captured_text not in self.molecules_set:
                        self.molecules_set.add(captured_text)
                except:
                    captured_text = ''
                out.append(captured_text)
            outputs.append(out)
        
        return outputs, raw_outputs
    
    def calculate_properties(self, property_fns):
        # TODO: drop the hard coded index and adjust for multiple targets
        calculated_properties = [property_fns[0](out) for out in self.outputs]

        return calculated_properties

    def get_stats(self):
        errors = []
        invalid_generations = 0
        for c_property, target in zip(self.calculated_properties, self.targets):
            error = []
            for prop in c_property:
                if prop != None:
                    error.append(abs(prop - target[0]))
                else:
                    error.append(0)
                    invalid_generations += 1
            errors.append(error)
        # errors.append(error)

        return errors, invalid_generations

    def write_to_file(self, test_name):
        self.log_file.write(f'properties under test: {test_name}\n')
        self.log_file.write(f'number of total generations: {len(self.inputs)}\n')
        self.log_file.write(f'number of unique molecules generated: {len(self.molecules_set)}\n')
        self.log_file.write(f"No valid SMILES generated in {self.invalid_generations} out of"\
                            f" {len(self.targets)} cases\n----------\n\n")

        for input, target, output, raw_output, c_prop, err in zip(self.inputs, self.targets,
                                                                  self.outputs, self.raw_outputs,
                                                                  self.calculated_properties,
                                                                  self.errors):
            self.log_file.write(f'input: {input}\n')
            self.log_file.write(f'target value: {target}\n')
            for r in raw_output:
                self.log_file.write(f'raw_output: {r}\n')
            for o, cp, e in zip(output, c_prop, err):
                self.log_file.write(f'captured_output: {o}\n')
                self.log_file.write(f'generated_property: {cp} diff: {e}\n')
            self.log_file.write('-----------\n')

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
        loss = np.sqrt(metrics.mean_squared_error(target_clean, generated_clean))
        return target_clean, generated_clean, nones, correlation, loss

    def generate_plot(self, test_name, target_clean, generated_clean, nones, correlation, loss):
        max_, min_ = np.max(self.targets), np.min(self.targets)
        title = f'greedy generation of {test_name} with {self.model_checkpoint_path.split("/")[-2]}\n rmse {loss:.3f}'
        if self.generation_config["do_sample"] == True:
            title = 'non ' + title       
        plt.title(title)
        plt.grid(True)
        plt.text(0.05 * max_, 0.90 * max(max(generated_clean), max_), f"Spearman correlation: {correlation:.3f}")
        plt.text(0.05 * max_, 0.85 * max(max(generated_clean), max_), f"N of Unique Mols: {len(self.molecules_set)}")
        plt.text(0.05 * max_, 0.80 * max(max(generated_clean), max_), f"N invalid gens: {self.invalid_generations}")
        plt.text(0.05 * max_, 0.75 * max(max(generated_clean), max_), f"N of total Gens: {len(self.inputs)}")
        plt.scatter(target_clean, generated_clean, c='b')
        plt.vlines(nones, ymin=min_, ymax=max_, color='r', alpha=0.3)
        plt.plot([min_, max_], [min_, max_], color='grey', linestyle='--', linewidth=2)
        plt.xlabel(f'target {test_name}')
        plt.ylabel(f'generated {test_name}')
        plt.savefig(self.results_path + test_name + '.png', dpi=300, format="png")
        plt.clf()

    def run_property_2_Mol_test(self):    
        for test_name, sample in list(self.test_suit.items()):
            time_start = time.time()
            input_properties = sample["input_properties"]
            target_properties = sample["target_properties"]
            self.targets = self.get_targets(target_properties)
            self.inputs = self.get_inputs(input_properties)
            property_fns = self.get_property_fns(target_properties)
            self.outputs, self.raw_outputs = self.generate_outputs()
            self.calculated_properties = self.calculate_properties(property_fns)
            self.errors, self.invalid_generations = self.get_stats()
            target_clean, generated_clean, nones, correlation, loss = self.clean_outputs()
            self.write_to_file(test_name)
            self.generate_plot(test_name, target_clean, generated_clean, nones, correlation, loss)
            print(f"finished evaluating test for {test_name}")
            print(f"{len(self.inputs)} samples evaluated in {time.time()-time_start} seconds")


if __name__ == "__main__":
    seed_value = 42
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    test_suit = {
        "sas": {
            "input_properties": ["sas"],
            "target_properties": ["sas"]
        },
        "qed": {
            "input_properties": ["qed"],
            "target_properties": ["qed"]
        },
        "weight": {
            "input_properties": ["weight"],
            "target_properties": ["weight"]
        },
        "clogp": {
            "input_properties": ["clogp"],
            "target_properties": ["clogp"]
        }
    }

    mock_test_suit = {
        "sas": {
            "input_properties": ["sas"],
            "target_properties": ["sas"]
        }
    }

    property_range = {
        "sas": {
            "range": (1, 10),
            "step":  0.1
        },
        "qed": {
            "range": (0, 1),
            "step":  0.01
        },
        "weight": {
            "range": (100, 500),
            "step":  0.5
        },
        "clogp": {
            "range": (1, 5),
            "step":  0.1
        }
    }  

    mock_property_range = {
        "sas": {
            "range": (1, 10),
            "step":  1
        }
    }

    greedy_generation_config = {
        "max_length": 300,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "do_sample": False,  
        "num_return_sequences": 1,
        "num_beams": 1
        }
    
    nongreedy_generation_config = {
        "max_length": 300,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "do_sample": True,  
        "num_return_sequences": 10,
        "num_beams": 1
        }
    
    regexp = "^.*?(?=\\[END_SMILES])"

    # model_checkpoint_path = "/home/hrant/chem/tigran/ChemLactica/checkpoints/facebook/galactica-125m/ac7915df73b24ee3a4e172d6/checkpoint-253952"
    model_125m_253k = "../checkpoints/125m_253k/"
    model_125m_241k = "../checkpoints/125m_241k/"
    model_1b_131k = "../checkpoints/1.3b_131k/"
    galactica_tokenizer_path = "./tokenizer/galactica-125m/"
    chemlactica_tokenizer_path = "./tokenizer/ChemLacticaTokenizer"
    # torch_dtype = float32
    torch_dtype = bfloat16
    device = "cuda:0"

    property_2_Mol = Property2Mol(
        test_suit=mock_test_suit,
        property_range=mock_property_range,
        generation_config=greedy_generation_config,
        regexp=regexp,
        model_checkpoint_path=model_125m_253k,
        tokenizer_path=chemlactica_tokenizer_path,
        torch_dtype=torch_dtype,
        device=device,
        )
    property_2_Mol.run_property_2_Mol_test()
    property_2_Mol.log_file.close()

##TODO: check for valid smiles
##      check for smiles in DB and train set
##      rank non greedy generations