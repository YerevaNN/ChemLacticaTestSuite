from transformers import OPTForCausalLM, AutoModelForCausalLM, AutoTokenizer
import torch
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def random_text(length):
    s = "abcdefghijklmnopqrstuvwxyz"
    inds = [random.randint(0, len(s) - 1) for _ in range(length)]
    return "".join([s[i] for i in inds])


if __name__ == "__main__":
    gen_batch_size = 100
    device = "cuda:0"
    # checkpoint_path = "yerevann/chemlactica-125m"
    # tokenizer_path = "yerevann/chemlactica-125m"
    checkpoint_path = "yerevann/chemlactica-1.3b"
    tokenizer_path = "yerevann/chemlactica-1.3b"
    # checkpoint_path = "yerevann/chemma-2b"
    # tokenizer_path = "yerevann/chemma-2b"
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
    # print(tokenizer)

    random_txt = random_text(100000)
    texts = [random_txt for _ in range(gen_batch_size)]
    data = tokenizer(texts, padding=True, return_tensors="pt").to(device)

    if type(model) == OPTForCausalLM:
        del data["token_type_ids"]
    for key, value in data.items():
        data[key] = value[:, :210]
    print(data.input_ids.shape)
    outputs = model.generate(
        **data,
        do_sample=True,
        max_new_tokens=100,
        repetition_penalty=1.0,
        eos_token_id=20
    )
    print(outputs)
    print(torch.cuda.max_memory_allocated(device) / 1e9)