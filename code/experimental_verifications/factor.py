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
import pandas as pd


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
    
def build_prompt_ARC_Challenge(data, prompts):
    prompt_list = []
    question = data['question']
    choices = data['choices']
    answerKey = data['answerKey']

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

def build_prompt_CSQA(data, prompts):
    prompt_list = []
    question = data['question']
    choices = data['choices']
    answerKey = data['answerKey']

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

def build_prompt_MMLU(data, prompts):
    prompt_list = []
    question = data['question']
    choices = data['choices']
    answerKey = data['answer']
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

def build_prompt_OpenBookQA(data, prompts):
    prompt_list = []
    question = data['question_stem']
    choices = data['choices']
    answerKey = data['answerKey']

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

def load_data_from_json(dataset):
    records = []
    path = f"input/{dataset}/data_500.jsonl"
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
            prompt_list = build_prompt_ARC_Challenge(d, prompts)
        elif args.dataset == "CommonSenseQA":
            prompt_list = build_prompt_CSQA(d, prompts)
        elif args.dataset == "MMLU":
            prompt_list = build_prompt_MMLU(d, prompts)
        elif args.dataset == "OpenBookQA":
            prompt_list = build_prompt_OpenBookQA(d, prompts)

        eval_data_list.append(
            {
                "question_id": id,
                "prompt_list": prompt_list
            }
        )
    return eval_data_list

def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=True, cache_dir=args.cache_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, cache_dir=args.cache_path).to(device)
    model.eval()
    return model, tokenizer

def encode_ids_and_mask(tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs['input_ids'].to(device)     
    attention_mask = inputs['attention_mask'].to(device)     
    return input_ids, attention_mask

def calculate_logit(model, tokenizer, prompt):
    input_ids, attention_mask = encode_ids_and_mask(tokenizer, prompt)

    correct = input_ids[0][-1]
    input_ids = input_ids[0][:-1]
    attention_mask = attention_mask[0][:-1]

    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True
    )

    last_logits = outputs.logits[:, -1, :]
    last_logprobs = torch.log_softmax(last_logits, dim=-1)
    logit = last_logprobs[0, correct]
    return logit.item()

def main(args):
    eval_data_list = create_eval_data(args.dataset)
    print(len(eval_data_list))

    result_list = []
    with torch.no_grad():
        for eval_data in tqdm(eval_data_list, total=len(eval_data_list)):
            question_id = eval_data['question_id']
            prompt_list = eval_data['prompt_list']

            for item in prompt_list:
                prompt_key = item['prompt_key']
                prompt = item['prompt']
                logit = calculate_logit(model, tokenizer, prompt)
                result_list.append([question_id, prompt_key, logit])

    header = ["question_id", "prompt_id", "logit"]

    df = pd.DataFrame(result_list, columns=header)
    save_path = f"results/data_results/factors/{args.model_name_or_path}/{args.dataset}_logit.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"Results saved to {save_path}")


if __name__ == '__main__':
    args = arguments()
    model, tokenizer = load_model(args)
    args.model = model
    args.tokenizer = tokenizer
    main(args)
    pass