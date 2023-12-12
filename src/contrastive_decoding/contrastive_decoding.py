from typing import Dict, Any, Tuple, Optional
import torch
from custom_modeling_opt import CustomOPTForCausalLM
from utils.dataset_utils import get_tokenizer
from transformers.generation.utils import GreedySearchDecoderOnlyOutput, ModelOutput


def _extract_past_from_model_output(outputs: ModelOutput):
    past_key_values = None
    if "past_key_values" in outputs:
        past_key_values = outputs.past_key_values
    elif "mems" in outputs:
        past_key_values = outputs.mems
    elif "past_buckets_states" in outputs:
        past_key_values = outputs.past_buckets_states
    return past_key_values


def _update_model_kwargs_for_generation(
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = _extract_past_from_model_output(outputs)
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
        return model_kwargs

@torch.no_grad()
def contrast_logits(
        expert_logits: torch.Tensor,
        student_logits: torch.Tensor,
        adaptability_constant: float,
        student_coef: float,
        expert_temp: float,
        student_temp: float
    ):
    batch_size, context_length, vocab_size = expert_logits.shape
    expert_probs = (expert_logits / expert_temp).softmax(-1)
    student_probs = (student_logits / student_temp).softmax(-1)
    contrasted_probs = torch.empty_like(expert_probs)
    for i in range(batch_size):
        inds_to_ignore = expert_probs[i] < adaptability_constant * expert_probs[i].max(-1).values.view(context_length, 1)
        contrasted_probs[i] = expert_probs[i] / (1 if student_coef == 0.0 else student_coef * student_probs[i])
        contrasted_probs[i][inds_to_ignore] = float('-inf')
    contrasted_probs[contrasted_probs != float('-inf')] = torch.log(contrasted_probs[contrasted_probs != float('-inf')])
    return contrasted_probs


@torch.no_grad()
def contrastive_forward(
        input_ids: torch.Tensor,
        expert_lm,
        student_lm,
        expert_temp: float=1.0,
        student_temp: float=1.0,
        student_coef: float=1.0,
        adaptability_constant: float=0.0,
        **model_kwargs
    ):
    assert 0 < expert_temp <= 1
    assert 0 < student_temp <= 1
    assert 0 <= student_coef <= 1
    assert 0 <= adaptability_constant <= 1

    expert_lm.eval()
    student_lm.eval()
    # define attention mask
    model_kwargs["attention_mask"] = torch.ones(input_ids.shape[:2], dtype=torch.long, device=input_ids.device)
    model_inputs = expert_lm.prepare_inputs_for_generation(input_ids, **model_kwargs)
    
    # forward pass to get next token
    expert_lm_outputs = expert_lm(
        **model_inputs,
        return_dict=True
    )
    student_lm_outputs = student_lm(
        **model_inputs,
        return_dict=True
    )

    expert_lm_logits = expert_lm_outputs.logits
    student_lm_logits = student_lm_outputs.logits

    next_tokens_scores = contrast_logits(
        expert_lm_logits,
        student_lm_logits,
        adaptability_constant=adaptability_constant,
        student_coef=student_coef,
        expert_temp=expert_temp,
        student_temp=student_temp
    )

    return next_tokens_scores


@torch.no_grad()
def contrastive_generate(
        input_ids: torch.Tensor,
        expert_lm,
        student_lm,
        do_sample: bool=False,
        expert_temp: float=1.0,
        student_temp: float=1.0,
        max_length: int=20,
        num_return_sequences: int=1,
        num_beams: int=1,
        output_attentions: bool=False,
        output_hidden_states: bool=False,
        return_dict_in_generate: bool=False,
        output_scores: bool=False,
        student_coef: float=1.0,
        adaptability_constant: float=0.0,
        **model_kwargs
    ):
    assert 0 < expert_temp <= 1
    assert 0 < student_temp <= 1
    assert 0 <= student_coef <= 1
    assert 0 <= adaptability_constant <= 1

    expert_lm.eval()
    student_lm.eval()
    # define attention mask
    model_kwargs["attention_mask"] = torch.ones(input_ids.shape[:2], dtype=torch.long, device=input_ids.device)

    scores = () if (return_dict_in_generate and output_scores) else None

    if not do_sample:
        for i in range(max_length):
            model_inputs = expert_lm.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            expert_lm_outputs = expert_lm(
                **model_inputs,
                return_dict=True
            )
            student_lm_outputs = student_lm(
                **model_inputs,
                return_dict=True
            )

            expert_lm_logits = expert_lm_outputs.logits
            student_lm_logits = student_lm_outputs.logits

            next_tokens_scores = contrast_logits(
                expert_lm_logits,
                student_lm_logits,
                adaptability_constant=adaptability_constant,
                student_coef=student_coef,
                expert_temp=expert_temp,
                student_temp=student_temp
            )
            
            next_tokens_scores = next_tokens_scores[:, -1, :]

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                # if output_attentions:
                #     decoder_attentions += (outputs.attentions,)
                # if output_hidden_states:
                #     decoder_hidden_states += (outputs.hidden_states,)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = _update_model_kwargs_for_generation(
                expert_lm_outputs, model_kwargs
            )

        if return_dict_in_generate:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                # note that this is the contrasted log probabilities, not logits
                scores=scores
            )
        return input_ids
    else:
        raise Exception("Non greedy decodings not implemented")


if __name__ == "__main__":
    device = "cuda:1"
    student_path = "/home/menuab/code/checkpoints/fe31d8c5edfd4b93b72f1b60/125m_120k_fe31"
    expert_path = "/home/menuab/code/checkpoints/cf982665b6c04c83a310b97d/125m_313k_cf98"
    expert_lm = CustomOPTForCausalLM.from_pretrained(expert_path, use_flash_attn=True, torch_dtype=torch.bfloat16).to(device)
    student_lm = CustomOPTForCausalLM.from_pretrained(student_path, use_flash_attn=True, torch_dtype=torch.bfloat16).to(device)
    
    tokenizer = get_tokenizer()
    input_prompt = "</s>[SAS]1.20[/SAS][START_SMILES]"
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt").to(device)
    outputs = contrastive_generate(
        input_ids=input_ids,
        expert_lm=expert_lm,
        student_lm=student_lm,
        return_dict_in_generate=True,
        output_scores=True,
        use_cache=True,
        student_coef=1.0,
        adaptability_constant=0.1,
        max_length=100
    )
    scores = expert_lm.compute_transition_scores(
        sequences=outputs.sequences,
        scores=outputs.scores,
        normalize_logits=True
    )
    s_outputs = expert_lm.generate(
        input_ids=input_ids,
        return_dict_in_generate=True,
        output_scores=True,
        use_cache=True
    )
    print(tokenizer.decode(outputs.sequences[0]))
    print(tokenizer.decode(s_outputs.sequences[0]))
    print(scores)