import random
import gc
import math

from transformers.data.data_collator import default_data_collator
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from contrastive_decoding import contrastive_forward
from pprint import pprint

from datasets import load_dataset
from  dataset_utils import process_dataset
from custom_modeling_opt import CustomOPTForCausalLM

torch.random.manual_seed(42)
random.seed(42)
np.random.seed(42)

def get_eval_dataloader(
        eval_dataset: torch.utils.data.Dataset,
        eval_batch_size: int,
        dataloader_num_workers: int,
        dataloader_pin_memory: bool
    ):
    data_collator = default_data_collator

    dataloader_params = {
        "batch_size": eval_batch_size,
        "collate_fn": data_collator,
        "num_workers": dataloader_num_workers,
        "pin_memory": dataloader_pin_memory,
        "drop_last": True
    }

    return DataLoader(eval_dataset, **dataloader_params)


def run_hparam_search():
    device = "cuda:0"
    student_path = "/home/menuab/code/checkpoints/fe31d8c5edfd4b93b72f1b60/125m_120k_fe31"
    expert_path = "/home/menuab/code/checkpoints/cf982665b6c04c83a310b97d/125m_313k_cf98"
    expert_lm = CustomOPTForCausalLM.from_pretrained(expert_path, use_flash_attn=True, torch_dtype=torch.bfloat16).to(device)
    student_lm = CustomOPTForCausalLM.from_pretrained(student_path, use_flash_attn=True, torch_dtype=torch.bfloat16).to(device)
    
    # tokenizer = get_tokenizer()
    # input_prompt = "</s>"
    # input_ids = tokenizer.encode(input_prompt, return_tensors="pt").to(device)
    eval_dataset = load_dataset(
        "text", data_files={"validation": "/home/tigranfahradyan/single_data/valid/71409_start_small.jsonl"}, streaming=False
    )
    processed_eval_dataset = process_dataset(
        dataset=eval_dataset["validation"],
        train_config={"block_size": 2048},
        process_batch_sizes=(50, 50),
        is_eval=False,
    )
    batch_size = 12
    eval_dataloader = get_eval_dataloader(processed_eval_dataset, batch_size, 8, True)
    ce_loss = torch.nn.CrossEntropyLoss()
    # print(len(eval_dataloader))

    print("eval dataloader size", len(eval_dataloader))
    grid_search_params = []
    for adapt_const in np.linspace(0.1, 0.9, 5):
        for st_coef in np.linspace(0.2, 1, 5):
            for st_temp in np.linspace(0.2, 1, 5):
                grid_search_params.append({
                    "adaptability_constant": adapt_const,
                    "student_coef": st_coef,
                    "student_temp": st_temp,
                    "expert_temp": 1.0,
                })

    for i, params in enumerate(grid_search_params):
        agg_loss = 0.0
        pprint(f"Grid search with {params}")
        student_coef = params["student_coef"]
        adaptability_constant = params["adaptability_constant"]
        expert_temp = params["expert_temp"]
        student_temp = params["student_temp"]
        for batch in tqdm(eval_dataloader):
            del batch["token_type_ids"]
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            logits = contrastive_forward(
                **batch,
                expert_lm=expert_lm,
                student_lm=student_lm,
                expert_temp=expert_temp,
                student_temp=student_temp,
                student_coef=student_coef,
                adaptability_constant=adaptability_constant,
            )
            batch_size, context_length, vocab_size = logits.shape
            logits = logits.view(batch_size * context_length, vocab_size)
            # for numerical stability, set what is -inf to -100
            logits[logits == float('-inf')] = -20

            targets = batch["labels"].view(batch_size * context_length).to(device)
            loss = ce_loss(logits, targets).detach().item()
            if not math.isnan(loss):
                agg_loss += loss / batch_size
        pprint(f"end loss {agg_loss / len(eval_dataloader)}")
        grid_search_params[i]["loss"] = agg_loss / len(eval_dataloader)
    pprint(grid_search_params)


if __name__ == "__main__":
    run_hparam_search()
