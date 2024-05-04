import argparse
import os
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import transformers
import yaml
from chemlactica.utils.model_utils import load_model


def get_model_perp_on_text(model,tokenizer,text):
    device = "cuda"

    encodings = tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            max_length=500,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
    encoded_texts = encodings["input_ids"].to(device)
    attn_masks = encodings["attention_mask"]
    loss_fct = CrossEntropyLoss(reduction="none")
    encoded_batch = encoded_texts[:]
    labels = encoded_batch
    attn_mask = attn_masks[:].to(device)

    with torch.no_grad():
        out_logits = model(encoded_texts, attention_mask=attn_mask).logits

    shift_logits = out_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_attention_mask_batch = attn_mask[..., 1:].contiguous()
    perplexity_batch = torch.exp(
        (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
        / shift_attention_mask_batch.sum(1)
    )
    perplexity_batch = perplexity_batch.tolist()
    print("length of perplexity batch", len(perplexity_batch))

    if len(perplexity_batch) == 1:
        ppl = perplexity_batch[0]
    else: 
        assert False
    return ppl



    


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    for run in config["runs"]: 
        run_name = run["name"]
        model_path = run["model_path"]
        tokenizer_path = run["tokenizer_path"]
        eval_text_file_path = run["eval_text_file_path"]
        print(model_path, tokenizer_path, eval_text_file_path)

        model = load_model(model_path,dtype = torch.bfloat16).to("cuda")
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        curr_run_output_path = os.path.join(config["results_path"], run_name)
        curr_run_log_path = os.path.join(curr_run_output_path ,f"{run_name}.log")
        if not os.path.exists(curr_run_output_path):
            os.makedirs(curr_run_output_path)

        with open(eval_text_file_path, 'r') as f:
             eval_text = f.readlines()
        eval_text = [x.strip() for x in eval_text]
        for test_text in eval_text:
            sample_result_dict = {}
            sample_result_dict["text"] = test_text
            ppl = get_model_perp_on_text(model,tokenizer,test_text)
            sample_result_dict["text"] = test_text
            sample_result_dict["ppl"] = ppl
            with open(curr_run_log_path, 'a+') as f:
                f.write(json.dumps(sample_result_dict) + '\n')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
