import gc
import json
import random
import time
from typing import List
import numpy as np
from tqdm import tqdm
import argparse
import torch
from vllm import LLM, SamplingParams
from utils import MultiProcessVllmInferencer

def format_inputs(positive_systems, negative_systems, instructions):
    original_inputs = []
    positive_inputs = []
    negative_inputs = []
    for p_system, n_system, instruction in zip(positive_systems, negative_systems, instructions):
        original_inputs.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True
        ))

        positive_inputs.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": p_system},{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True
        ))

        negative_inputs.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": n_system},{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True
        ))
    return original_inputs, positive_inputs, negative_inputs

def get_ppl(prompts: List[str], responses: List[str]) -> List[float]:
    batch_size = len(prompts)
    sampling_kwargs = SamplingParams(prompt_logprobs=2, skip_special_tokens=True, max_tokens=1)
    inputs_ids_lens = [len(tokenizer.encode(prompt)) for prompt in prompts]
    prompts = [prompt + response for prompt, response in zip(prompts, responses)]
    # forward
    outputs = inferencer.inference(prompts, sampling_kwargs, use_tqdm=False)
    # compute ppl
    ppls = []
    for i in range(batch_size):
        start_idx = inputs_ids_lens[i]
        prompt_logprobs = outputs[i].prompt_logprobs[(start_idx):]
        prompt_token_ids = outputs[i].prompt_token_ids[(start_idx):]
        prompt_logprobs_list = [
            prompt_logprobs[j][prompt_token_ids[j]]
            for j in range(len(prompt_logprobs))
        ]
        prompt_logprobs_list = [j.logprob for j in prompt_logprobs_list]
        prompt_logprobs_list = np.array(prompt_logprobs_list)
        ppl = prompt_logprobs_list.sum(axis=-1) / len(prompt_token_ids)
        ppls.append(ppl)
    return np.array(ppls)

def get_ppl_score(datas):
    positive_systems = [data['p_system'] for data in datas]
    negative_systems = [data['n_system'] for data in datas]
    instructions = [data['instruction'] for data in datas]
    raw = [data['raw'] for data in datas]
    chosens = [data['chosen'] for data in datas]
    rejecteds = [data['rejected'] for data in datas]
    original_inputs, positive_inputs, negative_inputs = format_inputs(positive_systems, negative_systems, instructions)
    o_chosen_ppl = get_ppl(original_inputs, chosens)
    o_rejected_ppl = get_ppl(original_inputs, rejecteds)
    p_chosen_ppl = get_ppl(positive_inputs, chosens)
    p_rejected_ppl = get_ppl(positive_inputs, rejecteds)
    n_chosen_ppl = get_ppl(negative_inputs, chosens)
    n_rejected_ppl = get_ppl(negative_inputs, rejecteds)
    for data, o_chosen, o_rejected, p_chosen, p_rejected, n_chosen, n_rejected in zip(datas, o_chosen_ppl, o_rejected_ppl, p_chosen_ppl, p_rejected_ppl, n_chosen_ppl, n_rejected_ppl):
        data['o_chosen_ppl'] = o_chosen
        data['o_rejected_ppl'] = o_rejected
        data['p_chosen_ppl'] = p_chosen
        data['p_rejected_ppl'] = p_rejected
        data['n_chosen_ppl'] = n_chosen
        data['n_rejected_ppl'] = n_rejected
    return datas

def filter(rawdatas):
    datas = get_ppl_score(rawdatas)
    filtered_datas = []

    for data in datas:
        if data['p_chosen_ppl'] > data['o_chosen_ppl'] and data['o_chosen_ppl'] > data['n_chosen_ppl'] and data['n_rejected_ppl'] > data['o_rejected_ppl'] and data['o_rejected_ppl'] > data['p_rejected_ppl']:
            filtered_datas.append(data)
    datas = filtered_datas

    # 1. 删除非near-onpolicy的数据 
    c_avg_ppl = sum([data['o_chosen_ppl'] for data in datas])/len(datas)
    n_avg_ppl = sum([data['o_rejected_ppl'] for data in datas])/len(datas)
    datas = [data for data in datas if data['o_chosen_ppl'] >= c_avg_ppl and data['o_rejected_ppl'] >= n_avg_ppl]

    # 2. 删除chosen和rejected较差的数据
    c_avg_diff = sum([data['p_chosen_ppl']-data['n_chosen_ppl'] for data in datas])/len(datas)
    n_avg_diff = sum([data['n_rejected_ppl']-data['p_rejected_ppl'] for data in datas])/len(datas)
    datas = [data for data in datas if data['p_chosen_ppl']-data['n_chosen_ppl'] > c_avg_diff and data['n_rejected_ppl']-data['p_rejected_ppl'] > n_avg_diff]

    return datas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    model = args.model
    datapath = args.datapath
    batch_size = args.batch_size
    print(f"model: {model}\ndatapath: {datapath}\nbatch_size: {batch_size}")
    with open(datapath, 'r') as f:
        datas = json.load(f)

    inferencer = MultiProcessVllmInferencer(model_path=model,
        num_gpus_per_model=1
    )
    tokenizer = inferencer.get_tokenizer()
    
    filter_data = filter(datas) 

    with open(datapath.replace(".json",f"_filter.json"), 'w') as f:
        json.dump(filter_data, f, indent=4)
