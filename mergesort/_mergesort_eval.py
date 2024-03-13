import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import datasets
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import itertools
import ast
import re
import argparse
import json
import math

def init_model(lora_dir):
    model = AutoModelForCausalLM.from_pretrained('huggyllama/llama-7b',cache_dir='/scratch/kyle/LLMDivideAndConquer/cache', torch_dtype=torch.float16, device_map='auto')
    model = PeftModel.from_pretrained(model, lora_dir)
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def generate(model, tokenizer, prompt, max_length):
    input_tok=tokenizer([prompt],add_special_tokens=False,padding=True)
    input_ids=torch.LongTensor(input_tok['input_ids']).cuda()
    input_length = input_ids.shape[1]
    attention_mask=torch.LongTensor(input_tok['attention_mask']).cuda()
    tokenized_samples = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length, 
        do_sample=False, 
        pad_token_id=tokenizer.pad_token_id
    )
    generated_text = tokenizer.batch_decode(tokenized_samples,skip_special_tokens=True)[0]
    return generated_text


def recursive_generate(model, tokenizer, prompt, max_length, max_depth, logs = []):
    if max_depth == 0:
        raise ValueError("Max depth reached")
    output = generate(model, tokenizer, prompt, max_length)
    sort_matches = re.findall(r'sort\(\[[^\]]+\]\)', output)
    if len(sort_matches)<2:
        logs.append(output)
        return output, logs
    for match in sort_matches[1:]:
        sorted_result = recursive_generate(model, tokenizer, f"{match} = ", max_length, max_depth-1)
        output = output.replace(match, parse_last_list(sorted_result))
    output = generate(model, tokenizer, f"{output} = ", max_length)
    logs.append(output)
    return output, logs


def parse_last_list(s):
    matches =  re.findall(r'\[[^\]]*\]', s)
    return matches[-1] if matches else None
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str)
    parser.add_argument("--style", type=str)
    parser.add_argument("--lora_dir", type=str)
    args = parser.parse_args()
    
    model, tokenizer = init_model(args.lora_dir)
    
    with open(f"mergesort/mergesort_{args.split}_{args.style}.json", 'r') as f:
        data = json.load(f)
    
    if args.style == "baseline" or args.style == "scratchpad":
        for i in data.keys():
            count = 0
            for j in range(len(data[i])):
                pred = generate(model, tokenizer, f"{data[i][j]['input']}", 2048).split("=")[-1].strip().rstrip()
                gold = data[i][j]['output']
                if pred == gold:
                    count += 1
            print(f"Num Correct length {i} elements: {count}")
            
    elif args.style == "recursive":
        for i in data.keys():
            count = 0
            for j in range(len(data[i])):
                pred = recursive_generate(model, tokenizer, f"{data[i][j]['input']}", 2048, math.ceil(math.log2(int(i))) + 1)[0].split("=")[-1].strip().rstrip()
                gold = data[i][j]['output']
                if pred == gold:
                    count += 1
            print(f"Num Correct length {i} elements: {count}")
    
    
    
    

