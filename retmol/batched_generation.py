from transformers import OPTForCausalLM, AutoTokenizer
import string
import torch
import random


if __name__ == "__main__":

    device = "cuda:0"
    tokenizer_path = "./chemlactica/ChemLacticaTokenizer66/"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
    checkpoint_path = "/nfs/dgx/raid/chem/checkpoints/facebook/galactica-125m/26d322857a184fcbafda5d4a/checkpoint-118784"
    model = OPTForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16).to(device)
    # data = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_ids = torch.randint(0, 50000, (400, 500), device=device)
    output = model.generate(
        input_ids,
        do_sample=True,
        max_new_tokens=50,
        repetition_penalty=1.0,
        # diversity_penalty=1.0,
        eos_token_id=20,
        temperature=1.5
    )
    output_texts = tokenizer.batch_decode(output)
    print(torch.cuda.max_memory_allocated(device) / 1e9, "gb max memory allocated")