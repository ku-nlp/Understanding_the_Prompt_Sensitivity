import os
import csv
import json
import numpy as np

from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
import torch
import os

os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""
download_dir = ""

from huggingface_hub import login

token = ""
login(token=token)
device = "cuda" if torch.cuda.is_available() else "cpu"

def arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path',
                        type=str, default="Qwen/Qwen1.5-0.5B")
    parser.add_argument('--target_token',
                        type=str, default="correctKey",
                        choices=["correctKey", "incorrectKey", "numberKey"])                   
    parser.add_argument('--dataset', type=str, default="OpenBookQA", 
                        choices=["ARC_Challenge", "CommonSenseQA", "MMLU", "OpenBookQA"])
    parser.add_argument(
        '--cache_path', default='')
    args = parser.parse_args()
    return args

def load_prompts(dataset):
    with open("input/12prompts.json", "r") as f:
        prompts = json.load(f)
    if dataset == "ARC_Challenge":
        return prompts["ARC_Challenge"]
    elif dataset == "CommonSenseQA":
        return prompts["CommonSenseQA"]
    elif dataset == "MMLU":
        return prompts["MMLU"]
    elif dataset == "OpenBookQA":
        return prompts["OpenBookQA"]
    else:
        raise NotImplementedError


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=True, cache_dir=args.cache_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, cache_dir=args.cache_path).to(device)
    model.eval()
    return model, tokenizer

def build_prompt_ARC_Challenge(target_token, data, prompts):
    prompt_list = []
    question = data['question']
    choices = data['choices']
    answerKey = data[target_token]

    for prompt_key, prompt_template in prompts.items():
        choices_text = choices['text']
        if len(choices_text) != 4:
            continue
        prompt = prompt_template.format(question=question,textA=choices_text[0],textB=choices_text[1],textC=choices_text[2],textD=choices_text[3])
        prompt = prompt + " " + answerKey
        prompt_list.append({
            "prompt_key": prompt_key,
            "prompt": prompt
        })
    return prompt_list

def build_prompt_CSQA(target_token, data, prompts):
    prompt_list = []
    question = data['question']
    choices = data['choices']
    answerKey = data[target_token]

    for prompt_key, prompt_template in prompts.items():
        choices_text = choices['text']
        if len(choices_text) != 5:
            continue
        prompt = prompt_template.format(question=question,textA=choices_text[0],textB=choices_text[1],textC=choices_text[2],textD=choices_text[3],textE=choices_text[4])
        prompt = prompt + " " + answerKey
        prompt_list.append({
            "prompt_key": prompt_key,
            "prompt": prompt
        })
    return prompt_list

def build_prompt_MMLU(target_token, data, prompts):
    prompt_list = []
    question = data['question']
    choices = data['choices']
    answerKey = data[target_token]
    for prompt_key, prompt_template in prompts.items():
        choices_text = choices
        if len(choices_text) != 4:
            continue
        prompt = prompt_template.format(question=question,textA=choices_text[0],textB=choices_text[1],textC=choices_text[2],textD=choices_text[3])
        prompt = prompt + " " + answerKey
        prompt_list.append({
            "prompt_key": prompt_key,
            "prompt": prompt
        })
    return prompt_list

def build_prompt_OpenBookQA(target_token, data, prompts):
    prompt_list = []
    question = data['question_stem']
    choices = data['choices']
    answerKey = data[target_token]

    for prompt_key, prompt_template in prompts.items():
        choices_text = choices['text']
        if len(choices_text) != 4:
            continue
        prompt = prompt_template.format(question=question,textA=choices_text[0],textB=choices_text[1],textC=choices_text[2],textD=choices_text[3])
        prompt = prompt + " " + answerKey
        prompt_list.append({
            "prompt_key": prompt_key,
            "prompt": prompt
        })
    return prompt_list

def encode_ids_and_mask(tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs['input_ids'].to(device)     
    attention_mask = inputs['attention_mask'].to(device)     
    return input_ids, attention_mask

def saliency(model, input_ids, attention_mask):
    torch.enable_grad()

    correct_id = input_ids[0][-1]
    input_ids = input_ids[0][:-1]
    attention_mask = attention_mask[0][:-1]

    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)

    model.zero_grad(set_to_none=True)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,           
        return_dict=True
    )

    hidden_states = list(outputs.hidden_states)  # tuple -> list
    for h in hidden_states:
        h.retain_grad()

    last_logits = outputs.logits[:, -1, :]
    last_logprobs = torch.log_softmax(last_logits, dim=-1)
    score = last_logprobs[0, correct_id]

    score.backward()

    log_prob = score.item()

    grads = [
        h.grad.detach()[:,-1,:].reshape(-1) if (h.grad is not None) else None
        for h in hidden_states
    ]
    output_hidden_states = [
            h.detach()[:,-1,:].squeeze(0)
            for h in hidden_states
        ]
    return log_prob, grads, output_hidden_states

def frobenius_norm(grads, length_norm=False):
    if grads is None:
        return 0.0
    if isinstance(grads, (list, tuple)):
        return [frobenius_norm(g, length_norm=length_norm) for g in grads]
    if torch.is_tensor(grads):
        flat = grads.reshape(-1)
        norm = torch.linalg.vector_norm(flat, ord=2)
        if length_norm and grads.ndim > 0:
            norm = norm / grads.shape[0]
        return norm
    arr = np.asarray(grads)
    norm = np.linalg.norm(arr.reshape(-1), ord=2)
    if length_norm and arr.ndim > 0:
        norm = norm / arr.shape[0]
    return float(norm)



def forward_score_from_ids(model, input_ids, input_mask, correct_id=None):
    device = next(model.parameters()).device
    if correct_id is None:
        correct_id = input_ids[-1]
    # prefix only
    ids = input_ids[:-1]
    mask = input_mask[:-1]
    ids_t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    mask_t = torch.tensor(mask, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(
            ids_t, 
            attention_mask=mask_t,
            output_hidden_states=True
        )
        last_logits = outputs.logits[:, -1, :]  # [1, V]
        last_logprobs = torch.log_softmax(last_logits, dim=-1)
        logit = last_logprobs[0, correct_id].item()
        # logit = outputs.logits[:, -1, correct_id].item()
    hidden_states = hidden_states = [
            h.detach().squeeze(0)
            for h in outputs.hidden_states
        ]
    return logit, hidden_states

def calculate_gradient(model, tokenizer, prompt):
    input_ids, attention_mask = encode_ids_and_mask(tokenizer, prompt)
    log_prob, layer_grads, hidden_states = saliency(model, input_ids, attention_mask)
    grads_norms = frobenius_norm(layer_grads, length_norm=False)
    grads_norms_length_norm = frobenius_norm(layer_grads, length_norm=True)

    return {
        "input_ids": input_ids.squeeze(), 
        "attention_mask": attention_mask.squeeze(), 
        "grads": layer_grads, # [layer, dim]
        "grads_norms": grads_norms, # [layer]
        "grads_norms_length_norm": grads_norms_length_norm, # [layer]
        "log_prob": log_prob, # single position
        "hidden_states": hidden_states # [layer, dim]
    }

def pad_to_len(arr, target_len):
    if torch.is_tensor(arr):
        cur_len = arr.shape[0]
        if cur_len >= target_len:
            return arr[:target_len, ...]
        pad_len = target_len - cur_len
        pad_shape = (pad_len,) + arr.shape[1:]
        padding = torch.zeros(pad_shape, dtype=arr.dtype, device=arr.device)
        return torch.cat([arr, padding], dim=0)
    cur_len, dim = arr.shape
    if cur_len >= target_len:
        return arr[:target_len, :]
    pad_len = target_len - cur_len
    padding = np.zeros((pad_len, dim), dtype=arr.dtype)
    return np.vstack([arr, padding])

def load_data_from_json(dataset):
    records = []
    path = f"input/target_token/{dataset}/data_500.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(len(records))
    return records

def create_eval_data(dataset):
    datas = load_data_from_json(dataset)
    eval_data_list = []   
    prompts = load_prompts(dataset)     
    for index, d in enumerate(datas):
        if d.get("id") is None:
            id = index
        else:
            id = d['id']

        if args.dataset == "ARC_Challenge":
            prompt_list = build_prompt_ARC_Challenge(args.target_token, d, prompts)
        elif args.dataset == "CommonSenseQA":
            prompt_list = build_prompt_CSQA(args.target_token, d, prompts)
        elif args.dataset == "MMLU":
            prompt_list = build_prompt_MMLU(args.target_token, d, prompts)
        elif args.dataset == "OpenBookQA":
            prompt_list = build_prompt_OpenBookQA(args.target_token, d, prompts)

        eval_data_list.append(
            {
                "question_id": id,
                "prompt_list": prompt_list
            }
        )
    return eval_data_list

def main(args):
    eval_data_list = create_eval_data(args.dataset)
    print(len(eval_data_list))

    result_list = []
    for eval_data in tqdm(eval_data_list, total=len(eval_data_list)):
        question_id = eval_data['question_id']
        prompt_list = eval_data['prompt_list']

        group_result_list = []
        for item in prompt_list:
            prompt_key = item['prompt_key']
            prompt = item['prompt']
            result = calculate_gradient(model, tokenizer, prompt)
            group_result_list.append(
                {
                    "prompt_key": prompt_key,
                    "result": result
                }
            )
       
        for i, rst0 in enumerate(group_result_list):
            prompt_key0 = rst0['prompt_key']
            result0 = rst0['result']
            for rst1 in group_result_list[i+1:]:
                prompt_key1 = rst1['prompt_key']
                result1 = rst1['result']
                if prompt_key0 == prompt_key1: continue

                delta_log_prob = result0['log_prob'] - result1['log_prob']
               
                hidden_states0 = result0['hidden_states']
                hidden_states1 = result1['hidden_states']
                
                delta_z_list = [h0 - h1 for h0, h1 in zip(hidden_states0, hidden_states1)]
                grad_list = result0['grads']
                # [layer, 1] |grad| * |delta z|
                linear_approx = [torch.sum(grad * delta) for grad, delta in zip(grad_list, delta_z_list)]
                delta_z_norm_list = [frobenius_norm(delta_z, length_norm=False) for delta_z in delta_z_list]

                grads_norem_list = [float(grad_norm.detach().cpu().numpy()) for grad_norm in result0['grads_norms']]
                grads_length_norm = [float(grad_length_norm.detach().cpu().numpy()) for grad_length_norm in result0['grads_norms_length_norm']]
                linear_approx_cpu = [float(lp.detach().cpu().numpy()) for lp in linear_approx]
                delta_z_norm_list_cpu = [float(delta_z_norm.detach().cpu().numpy()) for delta_z_norm in delta_z_norm_list]

                result_list.append(
                    {   
                        "question_id": question_id, # question_id
                        "prompt_key0": prompt_key0, # prompt_key0
                        "prompt_key1": prompt_key1, # prompt_key1
                        "delta_log_prob": delta_log_prob, # delta_log_prob
                        "delta_log_prob_norm": np.abs(delta_log_prob), # delta_log_prob
                        "grads_norm_list": grads_norem_list, # grad norm
                        "grads_length_norm": grads_length_norm, # grad_length_norm
                        "linear_approx": linear_approx_cpu,
                        "delta_z_norm_list": delta_z_norm_list_cpu
                    }
                )

    save_path = f"results/data_results/target_token/{args.target_token}/{args.model_name_or_path}/{args.dataset}_result.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for item in result_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    args = arguments()
    model, tokenizer = load_model(args)
    args.model = model
    args.tokenizer = tokenizer
    main(args)
