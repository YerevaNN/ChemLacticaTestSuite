from transformers import OPTForCausalLM, AutoModelForCausalLM, AutoTokenizer
import torch
import random


def random_text(length):
    s = "abcdefghijklmnopqrstuvwxyz"
    inds = [random.randint(0, len(s) - 1) for _ in range(length)]
    return "".join([s[i] for i in inds])


if __name__ == "__main__":
    gen_batch_size = 100
    device = "cuda:0"
    # checkpoint_path = "/nfs/dgx/raid/chem/checkpoints/facebook/galactica-125m/9954e52e400b43d18d3a40f6/checkpoint-20480"
    # tokenizer_path = "/auto/home/tigranfahradyan/RetMol/RetMol/chemlactica/ChemLacticaTokenizer66"
    # checkpoint_path = "/nfs/dgx/raid/chem/checkpoints/google/gemma-2b/869e097219da4a4fbbadcc11/checkpoint-20000"
    # tokenizer_path = "/auto/home/menuab/code/ChemLactica/chemlactica/tokenizer/GemmaTokenizer"

    checkpoint_path = "/home/admin/checkpoints/facebook/galactica-1.3b/6d68b252d53647a99cf2fa8b/last"
    tokenizer_path = "/home/admin/tigran/ChemLactica/chemlactica/tokenizer/ChemLacticaTokenizer66"
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
    # print(tokenizer)

    texts = [random_text(100000) for _ in range(gen_batch_size)]
    data = tokenizer(texts, padding=True, return_tensors="pt").to(device)

    if type(model) == OPTForCausalLM:
        del data["token_type_ids"]
    for key, value in data.items():
        data[key] = value[:, :1200]
    print(data.input_ids.shape)
    outputs = model.generate(
        **data,
        do_sample=True,
        max_new_tokens=100
    )
    print(outputs.shape)
    print(f"{torch.cuda.max_memory_allocated() / 1e9:.4f}Gb")