"""Conditional contrastive text generation using GPT like models"""


import logging
import sys 
import numpy as np
import torch
from tqdm import tqdm 
from contrastive_decodable_transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import json 
from collections import Counter

from contrastive_decodable_transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    OPTModel,
    OPTForCausalLM,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast
    # OPTLMHeadModel,
)

class NgramModel(torch.nn.Module):
    def __init__(self, n, vocab_size):
        super().__init__()
        self.n = n 
        self.alpha = 0.1 
        self.vocab_size = vocab_size

        
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        return {
            "input_ids": input_ids,
        }
    
    
    @staticmethod
    def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
        return generated_ngrams

    @staticmethod
    def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - ngram_size
        ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
        return banned_ngrams.get(ngram_idx, [])

    @staticmethod
    def _calc_banned_ngram_tokens(
        ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int
    ):
        """Copied from fairseq for no_repeat_ngram in beam_search"""
        if cur_len + 1 < ngram_size:
            # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            return [[] for _ in range(num_hypos)]

        generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)

        banned_tokens = [
            _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
            for hypo_idx in range(num_hypos)
        ]
        return banned_token

    def forward(self, input_ids, **kwargs):
        # print(input_ids)
        generated_ngrams = self._get_ngrams(self.n, input_ids, len(input_ids))
        base_lst = []
        for hypo_idx in range(len(input_ids)):
            ngram_cands = self._get_generated_ngrams(generated_ngrams[hypo_idx], input_ids[hypo_idx], self.n, input_ids.size(1))
            ngram_count = Counter(ngram_cands)
            # print(ngram_count, 'the number of occurences for different indices. ') 
            k_lst = list(ngram_count.keys())
            v_lst = list(ngram_count.values())
            k_lst = torch.LongTensor(k_lst).cuda()
            v_lst = torch.Tensor(v_lst).cuda()
            base = torch.ones(self.vocab_size).cuda() * self.alpha 
            base.scatter_add_(-1, k_lst, v_lst) 
            base_lst.append(base) 
        base_lst = torch.stack(base_lst, dim=0)
        # print(base_lst.shape)
        # normalize. 
        base_lst = base_lst / base_lst.sum(dim=-1).unsqueeze(-1)
        base_lst = base_lst.log().unsqueeze(1)
        # print(base_lst.shape, base_lst)
        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=base_lst,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None, 
        )

def analysis(model, generated, prompt_len):
    output = model(generated, labels=generated)
    print(output.loss)
    return 

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "opt": (OPTForCausalLM, PreTrainedTokenizerFast),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
    "gptj": (AutoModelForCausalLM, AutoTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""




def prepare_ctrl_input( _, tokenizer, prompt_text):
    if temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input( model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if xlm_language in available_languages:
            language = xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input( _, tokenizer, prompt_text):
    prefix = prefix if prefix else padding_text if padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input( _, tokenizer, prompt_text):
    prefix = prefix if prefix else padding_text if padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text

def opt_prepare_inputs_for_generation(input_ids, past=None, attention_mask=None, use_cache=None, **kwargs):
    if kwargs.get('useprompt', None):
        kwargs['useprompt'] = False
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }

    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    if past:
        input_ids = input_ids[:, -1:]
    # first step, decoder_cached_states are empty
    return {
        "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
        "attention_mask": attention_mask,
        "past_key_values": past,
        "use_cache": use_cache,
    }

# get_len = 31 
def ignore_prefix_opt_prepare_inputs_for_generation(input_ids, past=None, attention_mask=None, use_cache=None, **kwargs):
    if past is None:
        input_ids = input_ids[:, -1:]
    else:
        # print(past[0][0].shape) 
        genlen = past[0][0].shape[2] 
        input_ids = input_ids[:, -(genlen + 1):]
    # print(input_ids.shape) 

    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)


    input_ids = input_ids[:, -1:]
    # print(attention_mask.shape, input_ids.shape, 'ignore_prefix') 
    # first step, decoder_cached_states are empty
    return {
        "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
        "attention_mask": attention_mask,
        "past_key_values": past,
        "use_cache": use_cache,
    }

def ignore_prefix_prepare_inputs_for_generation(input_ids, past=None, **kwargs):
            
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    input_ids = input_ids[:, -1].unsqueeze(-1)
    if token_type_ids is not None:
        token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        position_ids = None

    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

def our_prepare_inputs_for_generation(input_ids, past=None, **kwargs):
    
    if kwargs.get('useprompt', None):
        kwargs['useprompt'] = False
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        token_type_ids = kwargs.get("token_type_ids", None)
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past:
            position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        position_ids = None
    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def out_file(outfile_path, generation_lst):
    with open(outfile_path, 'w') as f:
        for kk in generation_lst:
            print(json.dumps(kk), file=f) 

    print(f'written to {outfile_path}')
    return 

def format_out(generated_text, prompt, generated_tokens, gold_ref=None):
    output = {
                'ended'      : False,
                'tokens'     : generated_tokens,
                'prompt'     : prompt,
                'gen_text'   : generated_text, 
                'len'        : 0,
                'nll4tok'    : [],
                'ppl4tok'    : [],
                'ppl'        : 0,
                'gold_ref'   : gold_ref, 
            } 
            
    return output


def generate(
        input_ids: torch.tensor,
        expert_lm,
        student_lm,
        do_sample: bool=False,
        length: int=20,
        temperature: float=1.0,
        repetition_penalty: float=1.0,
        num_beam: int=1,
        top_k: int=0,
        top_p: float=1.0,
        min_prob: float=0.0,
        student_min_prob: float=0.0,
        student_temperature: float=1.0,
        num_return_sequences=1,
        contrastive_decoding: str="student",
        device="cpu",

        # not important for use
        use_cap_student="no",
        ignore_prefix="yes",
        use_switch="no",
        prefix="",
        padding_text="",
        prompt="",
        contrastive_prompt="",
        st_coef: float=0.5,
        stop_token=None,
        fp16=False,

        # not used
        # xlm_language',
        # seed,
        # no_cuda,
        # revision="checkpoint-20000",
        **generate_kwargs
    ):
    assert contrastive_decoding == "student"
    # if not do_sample and (contrastive_decoding == 'student' or contrastive_decoding == 'earlystop' or contrastive_decoding == 'ngram') :
    args = {
        "input_ids": input_ids,
        "max_length": length + len(input_ids),
        "min_length": length + len(input_ids),
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "min_prob": min_prob,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
        "num_beams": num_beam,
        "num_return_sequences": num_return_sequences,
        # "student_lm": student_lm,
        "teacher_student": True,
        "model_kwargs_student": {}, 
        "st_coef": st_coef,
        "tokenizer": tokenizer, # analysis
        "student_min_prob": student_min_prob,
        "student_temperature": student_temperature,
        "use_cap_student": (use_cap_student=='yes'), #cap student debug
        "use_switch": (use_switch == 'yes'),
    }
    print(args)
    output_sequences = expert_lm.generate(
        input_ids=input_ids,
        max_length=length + len(input_ids),
        min_length=length + len(input_ids),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_prob=min_prob,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        num_beams=num_beam,
        num_return_sequences=num_return_sequences,
        student_lm=student_lm,
        teacher_student=True,
        model_kwargs_student={}, 
        st_coef=st_coef,
        tokenizer=tokenizer, # analysis
        student_min_prob=student_min_prob,
        student_temperature=student_temperature,
        use_cap_student=(use_cap_student=='yes'), #cap student debug
        use_switch=(use_switch == 'yes'),
        **generate_kwargs
    )
    return output_sequences
    

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

if __name__ == "__main__":
    set_seed(42)
    device = "cuda:0"

    # tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-125m")
    tokenizer = AutoTokenizer.from_pretrained("/home/menuab/code/ChemLacticaTestSuite/src/tokenizer/ChemLacticaTokenizer_50028")

    # input_prompt = "</s> A version of Sonic the Hedgehog was developed by Ancient and released in 1991"
    input_prompt = "</s>[WEIGHT 875.1][START_SMILES]"
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt").to(device)

    # from transformers import AutoModelForCausalLM, OPTForCausalLM
    # gal_big = AutoModelForCausalLM.from_pretrained("facebook/galactica-1.3b")
    # print(tokenizer.decode(gal_big.generate(
    #     input_ids=input_ids,
    #     num_beams=5
    # )[0]))

                # load the expert model with OPTForCausalLM 
    # expert_lm = OPTForCausalLM.from_pretrained("facebook/galactica-1.3b").to(device)
    expert_lm = OPTForCausalLM.from_pretrained("/home/menuab/code/checkpoints/0d992caa5ec443d9aefc289c/125m_256k_0d99").to(device)
                # load the student model with AutoModelForCausalLM it is important
    # student_lm = AutoModelForCausalLM.from_pretrained("facebook/galactica-125m").to(device)
    student_lm = AutoModelForCausalLM.from_pretrained("/home/menuab/code/checkpoints/0d992caa5ec443d9aefc289c/125m_253k_0d99").to(device)

    # print(input_ids)
    output_tokens = generate(
        input_ids=input_ids,
        expert_lm=expert_lm,
        student_lm=student_lm,
        eos_token_id=20,
        # num_beam=1,
        return_dict_in_generate=True,
        output_scores=True,
        # length=256,
        st_coef=1.0,
        student_temperature=0.5,
    )
    # out = student_lm.generate(input_ids, max_length=200, do_sample=False)
    print(output_tokens)
    print(tokenizer.decode(output_tokens.sequences[0]))